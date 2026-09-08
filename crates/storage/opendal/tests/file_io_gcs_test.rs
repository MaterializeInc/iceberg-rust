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
    use reqsign_core::Context;

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

    /// Builds a `FileIO` whose only credential source should be `loader`, while also passing the
    /// static `gcs.oauth2.token` prop that a vending catalog would supply.
    async fn get_file_io_gcs_with_loader(loader: MockCredentialLoader) -> FileIO {
        set_up();

        let gcs_endpoint = get_gcs_endpoint();
        create_bucket(FAKE_GCS_BUCKET, &gcs_endpoint).await.unwrap();

        FileIOBuilder::new(Arc::new(OpenDalStorageFactory::Gcs {
            customized_credential_load: Some(CustomGcsCredentialLoader::new(loader)),
        }))
        .with_props(vec![
            (GCS_SERVICE_PATH, gcs_endpoint),
            (GCS_TOKEN, "stale-baked-in-token".to_string()),
        ])
        .build()
    }

    #[tokio::test]
    async fn test_gcs_with_custom_credential_loader() {
        let file_io = get_file_io_gcs_with_loader(MockCredentialLoader(Some(
            GcsCredential::with_token(GcsToken {
                access_token: "fresh-token".to_string(),
                expires_at: None,
            }),
        )))
        .await;

        file_io
            .exists(format!("{}/any", get_gs_path()))
            .await
            .expect("a loader that vends a credential signs successfully");
    }

    #[tokio::test]
    async fn test_gcs_custom_credential_loader_has_no_fallback() {
        // OpenDAL's GCS service *prepends* a custom chain to its own rather than replacing it,
        // and the chain swallows a provider's miss. So a loader that comes up empty must still
        // fail the request: falling through to the static `gcs.oauth2.token` above, which is
        // exactly the credential the loader exists to refresh, would leave an expired token in
        // force with no diagnostic.
        let file_io = get_file_io_gcs_with_loader(MockCredentialLoader(None)).await;

        let err = file_io
            .exists(format!("{}/any", get_gs_path()))
            .await
            .expect_err("a loader that vends nothing must not fall back");
        assert!(
            err.to_string()
                .contains("failed to load signing credential"),
            "unexpected error: {err}"
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
