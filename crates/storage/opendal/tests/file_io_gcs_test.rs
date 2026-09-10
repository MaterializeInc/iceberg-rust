// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Integration tests for FileIO Google Cloud Storage (GCS).
//!
//! These tests assume Docker containers are started externally via `make docker-up`.

#[cfg(feature = "opendal-gcs")]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use bytes::Bytes;
    use iceberg::io::{FileIO, FileIOBuilder, GCS_NO_AUTH, GCS_SERVICE_PATH, GCS_TOKEN};
    use iceberg_storage_opendal::{
        CustomGcsCredentialLoader, GcsCredential, GcsToken, OpenDalStorageFactory,
        ProvideCredential,
    };
    use iceberg_test_utils::{get_gcs_endpoint, set_up};
    use opendal::services::GcsConfig;
    use opendal::{Configurator, Operator};
    use reqsign_core::{Context, ProvideCredentialChain};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    static FAKE_GCS_BUCKET: &str = "test-bucket";

    async fn get_file_io_gcs() -> FileIO {
        set_up();

        let gcs_endpoint = get_gcs_endpoint();

        // A bucket must exist for FileIO
        create_bucket(FAKE_GCS_BUCKET, &gcs_endpoint).await.unwrap();

        FileIOBuilder::new(Arc::new(OpenDalStorageFactory::Gcs {
            customized_credential_load: None,
        }))
        .with_props(vec![
            (GCS_SERVICE_PATH, gcs_endpoint),
            (GCS_NO_AUTH, "true".to_string()),
        ])
        .build()
    }

    // Create a bucket against the emulated GCS storage server.
    async fn create_bucket(name: &str, server_endpoint: &str) -> anyhow::Result<()> {
        let mut bucket_data = HashMap::new();
        bucket_data.insert("name", name);

        let client = reqwest::Client::new();
        let endpoint = format!("{server_endpoint}/storage/v1/b");
        client.post(endpoint).json(&bucket_data).send().await?;
        Ok(())
    }

    fn get_gs_path() -> String {
        format!("gs://{FAKE_GCS_BUCKET}")
    }

    /// A loader that hands back whatever it was constructed with, standing in for a catalog that
    /// vends a credential or fails to.
    #[derive(Debug)]
    struct MockCredentialLoader(Option<GcsCredential>);

    impl ProvideCredential for MockCredentialLoader {
        type Credential = GcsCredential;

        async fn provide_credential(
            &self,
            _ctx: &Context,
        ) -> reqsign_core::Result<Option<GcsCredential>> {
            Ok(self.0.clone())
        }
    }

    /// Serves one request with a 404 and reports the `Authorization` header it saw, or `None` if
    /// the header was absent.
    ///
    /// fake-gcs-server ignores credentials outright, so it cannot witness *which* token signed a
    /// request. This stands in for it wherever that is the property under test. Returns the
    /// endpoint to point `gcs.service.path` at.
    async fn serve_one_request() -> (String, oneshot::Receiver<Option<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");

            // A GCS metadata read carries no body, so the header block is the whole request.
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                match socket.read(&mut buf).await.expect("read") {
                    0 => break,
                    n => request.extend_from_slice(&buf[..n]),
                }
            }
            socket
                .write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n")
                .await
                .expect("write response");

            let request = String::from_utf8_lossy(&request);
            let authorization = request
                .lines()
                .filter_map(|line| line.split_once(':'))
                .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.trim().to_string());
            let _ = tx.send(authorization);
        });

        (endpoint, rx)
    }

    /// Builds a `FileIO` whose only credential source should be `loader`, while also passing the
    /// static `gcs.oauth2.token` prop that a vending catalog would supply. That prop is the
    /// credential a loader exists to displace, so leaving it set is what makes the assertions
    /// below meaningful.
    fn file_io_with_loader(endpoint: String, loader: MockCredentialLoader) -> FileIO {
        FileIOBuilder::new(Arc::new(OpenDalStorageFactory::Gcs {
            customized_credential_load: Some(CustomGcsCredentialLoader::new(loader)),
        }))
        .with_props(vec![
            (GCS_SERVICE_PATH, endpoint),
            (GCS_TOKEN, "stale-baked-in-token".to_string()),
        ])
        .build()
    }

    #[tokio::test]
    async fn test_gcs_custom_credential_loader_signs_with_its_token() {
        let (endpoint, authorization) = serve_one_request().await;
        let file_io = file_io_with_loader(
            endpoint,
            MockCredentialLoader(Some(GcsCredential::with_token(GcsToken {
                access_token: "fresh-token".to_string(),
                expires_at: None,
            }))),
        );

        assert!(
            !file_io
                .exists(format!("{}/any", get_gs_path()))
                .await
                .expect("the loader's credential signs successfully"),
            "the stub server answers 404"
        );

        // The loader's token reached the wire, and the static prop did not.
        assert_eq!(
            authorization.await.expect("the stub server saw a request"),
            Some("Bearer fresh-token".to_string())
        );
    }

    #[tokio::test]
    async fn test_gcs_custom_credential_loader_has_no_fallback() {
        // OpenDAL's GCS service *prepends* a custom chain to its own rather than replacing it,
        // and the chain swallows a provider's miss. So a loader that comes up empty must still
        // fail the request: falling through to the static `gcs.oauth2.token` prop, which is
        // exactly the credential the loader exists to refresh, would leave an expired token in
        // force with no diagnostic.
        let (endpoint, mut authorization) = serve_one_request().await;
        let file_io = file_io_with_loader(endpoint, MockCredentialLoader(None));

        let err = file_io
            .exists(format!("{}/any", get_gs_path()))
            .await
            .expect_err("a loader that vends nothing must not fall back");
        assert!(
            err.to_string()
                .contains("failed to load signing credential"),
            "unexpected error: {err}"
        );
        // Signing fails before any I/O, so the stub server was never reached at all.
        assert_eq!(
            authorization.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        );
    }

    /// The control for [`test_gcs_custom_credential_loader_has_no_fallback`]: the same empty
    /// loader and the same stale config token, but skipping the suppression that
    /// `gcs_config_build` applies by manually constructing the credential chain. Here the request
    /// succeeds, signed with the credential the loader was supposed to displace.
    #[tokio::test]
    async fn test_opendal_gcs_chain_falls_through_to_the_config_token() {
        let (endpoint, authorization) = serve_one_request().await;

        // Disabling the ambient sources leaves the chain with exactly two entries that can
        // produce anything: the empty loader, then the token.
        let mut cfg = GcsConfig::default();
        cfg.bucket = FAKE_GCS_BUCKET.to_string();
        cfg.endpoint = Some(endpoint);
        cfg.token = Some("stale-baked-in-token".to_string());
        cfg.disable_vm_metadata = true;
        cfg.disable_config_load = true;

        let chain = ProvideCredentialChain::new().push(MockCredentialLoader(None));
        let operator = Operator::new(cfg.into_builder().credential_provider_chain(chain))
            .expect("operator builds");

        assert!(
            !operator.exists("any").await.expect("the request is signed"),
            "the stub server answers 404"
        );
        assert_eq!(
            authorization.await.expect("the stub server saw a request"),
            Some("Bearer stale-baked-in-token".to_string())
        );
    }

    #[tokio::test]
    async fn test_gcs_custom_credential_loader_rejects_unsigned_requests() {
        set_up();

        // `gcs.no-auth` skips signing entirely, which would silently sideline the loader.
        let file_io = FileIOBuilder::new(Arc::new(OpenDalStorageFactory::Gcs {
            customized_credential_load: Some(CustomGcsCredentialLoader::new(MockCredentialLoader(
                None,
            ))),
        }))
        .with_props(vec![
            (GCS_SERVICE_PATH, get_gcs_endpoint()),
            (GCS_NO_AUTH, "true".to_string()),
        ])
        .build();

        let err = file_io
            .exists(format!("{}/any", get_gs_path()))
            .await
            .expect_err("a loader combined with gcs.no-auth is a configuration error");
        assert!(
            err.to_string().contains("disable request signing"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn gcs_exists() {
        let file_io = get_file_io_gcs().await;
        assert!(file_io.exists(format!("{}/", get_gs_path())).await.unwrap());
    }

    #[tokio::test]
    async fn gcs_write() {
        let gs_file = format!("{}/write-file", get_gs_path());
        let file_io = get_file_io_gcs().await;
        let output = file_io.new_output(&gs_file).unwrap();
        output
            .write(bytes::Bytes::from_static(b"iceberg-gcs!"))
            .await
            .expect("Write to test output file");
        assert!(file_io.exists(gs_file).await.unwrap())
    }

    #[tokio::test]
    async fn gcs_read() {
        let gs_file = format!("{}/read-gcs", get_gs_path());
        let file_io = get_file_io_gcs().await;
        let output = file_io.new_output(&gs_file).unwrap();
        output
            .write(bytes::Bytes::from_static(b"iceberg!"))
            .await
            .expect("Write to test output file");
        assert!(file_io.exists(&gs_file).await.unwrap());

        let input = file_io.new_input(gs_file).unwrap();
        assert_eq!(input.read().await.unwrap(), Bytes::from_static(b"iceberg!"));
    }
}
