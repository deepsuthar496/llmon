use std::path::PathBuf;

use crate::branding;

/// Runtime configuration. Every user-visible default derives from `branding`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub models_dir: PathBuf,
    pub keep_alive_secs: u64,
    pub num_ctx: u32,
}

impl Config {
    pub fn from_env_or(host: Option<String>, port: Option<u16>) -> Self {
        let p = branding::ENV_PREFIX;
        let env = |k: &str| std::env::var(format!("{p}_{k}")).ok();
        let host = host
            .or_else(|| env("HOST"))
            .unwrap_or_else(|| branding::DEFAULT_HOST.to_string());
        let port = port
            .or_else(|| env("PORT").and_then(|v| v.parse().ok()))
            .unwrap_or(branding::DEFAULT_PORT);
        let models_dir = env("MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(default_models_dir);
        let keep_alive_secs = env("KEEP_ALIVE")
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);
        let num_ctx = env("NUM_CTX").and_then(|v| v.parse().ok()).unwrap_or(4096);
        Self {
            host,
            port,
            models_dir,
            keep_alive_secs,
            num_ctx,
        }
    }

    pub fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

pub fn default_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(branding::MODELS_ENV) {
        return PathBuf::from(dir);
    }
    match dirs::home_dir() {
        Some(h) => h.join(branding::HOME_DIRNAME).join("models"),
        None => PathBuf::from(format!("./{}", branding::HOME_DIRNAME)).join("models"),
    }
}
