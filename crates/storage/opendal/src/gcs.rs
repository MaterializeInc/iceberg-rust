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
//! Google Cloud Storage properties

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::io::{
    GCS_ALLOW_ANONYMOUS, GCS_CREDENTIALS_JSON, GCS_DISABLE_CONFIG_LOAD, GCS_DISABLE_VM_METADATA,
    GCS_NO_AUTH, GCS_SERVICE_PATH, GCS_TOKEN,
};
use iceberg::{Error, ErrorKind, Result};
use opendal::services::GcsConfig;
use opendal::{Configurator, Operator};
use reqsign_core::{ProvideCredential, ProvideCredentialChain, ProvideCredentialDyn};
/// GCS credentials: either a service account or an OAuth2 access token.
pub use reqsign_google::Credential as GcsCredential;
/// An OAuth2 access token, the form a catalog-vended GCS credential takes.
pub use reqsign_google::Token as GcsToken;
use url::Url;

use crate::utils::{from_opendal_error, is_truthy};

/// Parse iceberg properties to [`GcsConfig`].
pub(crate) fn gcs_config_parse(mut m: HashMap<String, String>) -> Result<GcsConfig> {
    let mut cfg = GcsConfig::default();

    if let Some(cred) = m.remove(GCS_CREDENTIALS_JSON) {
        cfg.credential = Some(cred);
    }

    if let Some(token) = m.remove(GCS_TOKEN) {
        cfg.token = Some(token);
    }

    if let Some(endpoint) = m.remove(GCS_SERVICE_PATH) {
        cfg.endpoint = Some(endpoint);
    }

    if m.remove(GCS_NO_AUTH).is_some() {
        cfg.skip_signature = true;
        cfg.disable_vm_metadata = true;
        cfg.disable_config_load = true;
    }

    if let Some(allow_anonymous) = m.remove(GCS_ALLOW_ANONYMOUS)
        && is_truthy(allow_anonymous.to_lowercase().as_str())
    {
        cfg.skip_signature = true;
    }
    if let Some(disable_ec2_metadata) = m.remove(GCS_DISABLE_VM_METADATA)
        && is_truthy(disable_ec2_metadata.to_lowercase().as_str())
    {
        cfg.disable_vm_metadata = true;
    };
    if let Some(disable_config_load) = m.remove(GCS_DISABLE_CONFIG_LOAD)
        && is_truthy(disable_config_load.to_lowercase().as_str())
    {
        cfg.disable_config_load = true;
    };

    Ok(cfg)
}

/// Clear every credential source opendal would consult on its own.
///
/// opendal's GCS service *prepends* a caller-supplied credential chain to its own
/// (unlike its S3 service, which replaces it), and [`ProvideCredentialChain`] swallows a
/// provider's error and falls through to the next entry. So without this, a custom loader
/// that fails is indistinguishable from one that was never installed: signing quietly
/// succeeds using whatever else the chain can reach, including the very static
/// `gcs.oauth2.token` the loader exists to replace, or ambient credentials on a GCE
/// instance.
fn suppress_default_credential_sources(cfg: &mut GcsConfig) {
    cfg.token = None;
    cfg.credential = None;
    cfg.credential_path = None;
    cfg.service_account = None;
    cfg.disable_vm_metadata = true;
    cfg.disable_config_load = true;
}

/// Build a new OpenDAL [`Operator`] based on a provided [`GcsConfig`].
pub(crate) fn gcs_config_build(
    cfg: &GcsConfig,
    customized_credential_load: &Option<CustomGcsCredentialLoader>,
    path: &str,
) -> Result<Operator> {
    let url = Url::parse(path)?;
    let bucket = url.host_str().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Invalid gcs url: {path}, bucket is required"),
        )
    })?;

    let mut cfg = cfg.clone();
    cfg.bucket = bucket.to_string();

    let builder = match customized_credential_load {
        None => cfg.into_builder(),
        Some(loader) => {
            // `skip_signature` bypasses the signer entirely, so the loader would never be
            // consulted. Refuse rather than silently ignore one of the two.
            if cfg.skip_signature {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "a custom GCS credential loader cannot be combined with \
                         {GCS_NO_AUTH} or {GCS_ALLOW_ANONYMOUS}, which disable request signing"
                    ),
                ));
            }
            suppress_default_credential_sources(&mut cfg);
            let chain = ProvideCredentialChain::new().push(Arc::clone(&loader.0));
            cfg.into_builder().credential_provider_chain(chain)
        }
    };

    Operator::new(builder).map_err(from_opendal_error)
}

/// Custom GCS credential loader.
///
/// Wraps any [`ProvideCredential`] implementation for use with the GCS storage backend.
/// Use [`CustomGcsCredentialLoader::new`] to create one, then pass it to
/// [`OpenDalStorageFactory::Gcs`](crate::OpenDalStorageFactory).
///
/// Installing a loader suppresses every credential source opendal would otherwise reach
/// for, so the loader is the sole authority on credentials. It must therefore succeed on
/// its own: there is no application default fallback behind it.
pub struct CustomGcsCredentialLoader(Arc<dyn ProvideCredentialDyn<Credential = GcsCredential>>);

impl Clone for CustomGcsCredentialLoader {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl std::fmt::Debug for CustomGcsCredentialLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomGcsCredentialLoader")
            .finish_non_exhaustive()
    }
}

impl CustomGcsCredentialLoader {
    /// Create a new custom GCS credential loader from any [`ProvideCredential`] implementation.
    pub fn new(provider: impl ProvideCredential<Credential = GcsCredential> + 'static) -> Self {
        Self(Arc::new(provider) as Arc<dyn ProvideCredentialDyn<Credential = GcsCredential>>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct NoopProvider;

    impl ProvideCredential for NoopProvider {
        type Credential = GcsCredential;

        async fn provide_credential(
            &self,
            _ctx: &reqsign_core::Context,
        ) -> reqsign_core::Result<Option<Self::Credential>> {
            Ok(None)
        }
    }

    #[test]
    fn loader_rejects_unsigned_requests() {
        let mut cfg = GcsConfig::default();
        cfg.skip_signature = true;
        let loader = Some(CustomGcsCredentialLoader::new(NoopProvider));

        let err = gcs_config_build(&cfg, &loader, "gs://bucket/key").unwrap_err();
        assert!(
            err.to_string().contains("disable request signing"),
            "unexpected error: {err}"
        );
    }
}
