//! Single rename point for the whole software.
//!
//! To rebrand (e.g. `llmon` -> `myapp`): change ONLY the constants below
//! (plus `[[bin]] name` in Cargo.toml). All CLI help, paths, env vars,
//! server headers and docs derive from here. Nothing else hardcodes the name.

/// User-visible application name.
pub const APP_NAME: &str = "llmon";
/// Binary / command name invoked by users.
pub const APP_BIN: &str = "llmon";
/// Version string (mirrors Cargo package version).
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
/// One-line tagline used in `--help` and server headers.
pub const APP_TAGLINE: &str = "fast, lightweight, single-binary LLM runner";
/// Default daemon port.
pub const DEFAULT_PORT: u16 = 11435;
/// Default daemon host.
pub const DEFAULT_HOST: &str = "127.0.0.1";
/// Env prefix, e.g. `LLMON_HOST`, `LLMON_PORT`, `LLMON_MODELS`.
pub const ENV_PREFIX: &str = "LLMON";
/// Home directory name, e.g. `~/.llmon`.
pub const HOME_DIRNAME: &str = ".llmon";
/// Env var overriding the model store root.
pub const MODELS_ENV: &str = "LLMON_MODELS";
/// HTTP server header value.
#[allow(dead_code)]
pub const SERVER_HEADER: &str = "llmon/0.1.0";
/// Default Hugging Face repo used by `pull <name>` shorthand.
pub const DEFAULT_HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
/// Ollama-compatible registry base (only used for `FROM registry:` lines
/// and `pull registry:...`).
pub const REGISTRY_BASE: &str = "https://registry.ollama.ai";

/// Generic alias so internal code never spells the product name directly.
#[allow(dead_code)]
pub fn app_name() -> &'static str {
    APP_NAME
}
