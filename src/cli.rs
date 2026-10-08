use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::branding;

#[derive(Parser, Debug)]
#[command(name = branding::APP_BIN, version = branding::APP_VERSION, about = format!("{} — {}", branding::APP_NAME, branding::APP_TAGLINE))]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Option<Cmd>,
    /// Daemon host (serve/run).
    #[arg(long, env = "LLMON_HOST")]
    pub host: Option<String>,
    /// Daemon port (serve/run).
    #[arg(long, env = "LLMON_PORT")]
    pub port: Option<u16>,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Start the HTTP daemon (Ollama-native /api/* + OpenAI /v1/*).
    Serve {
        #[arg(long)] host: Option<String>,
        #[arg(long)] port: Option<u16>,
        /// Backend context size (overrides LLMON_CTX / Modelfile PARAMETER).
        #[arg(long)] ctx_size: Option<u32>,
        /// Backend inference threads (overrides LLMON_THREADS).
        #[arg(long)] threads: Option<u32>,
        /// Backend batch size (overrides LLMON_BATCH).
        #[arg(long)] batch_size: Option<u32>,
    },
    /// Chat with a model (starts embedded daemon client if needed).
    Run {
        model: String,
        /// Print timings (load, prompt eval, eval rate) after each reply.
        #[arg(long)] verbose: bool,
        /// Keep model loaded after the reply (e.g. 5m, 1h, 0 to unload).
        #[arg(long)] keepalive: Option<String>,
        /// Output format: "json" forces JSON-valid replies.
        #[arg(long)] format: Option<String>,
        /// Enable model-defined thinking with an optional level
        /// (true/false/high/medium/low).
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        think: Option<String>,
        /// Hide the model's thinking output when thinking is enabled.
        #[arg(long)] hidethinking: bool,
        /// Backend context size for this run (local spawns; a live daemon
        /// uses its own serve flags).
        #[arg(long)] ctx_size: Option<u32>,
        /// Backend inference threads for this run (local spawns only).
        #[arg(long)] threads: Option<u32>,
        /// Backend batch size for this run (local spawns only).
        #[arg(long)] batch_size: Option<u32>,
        /// Draft model for speculative decoding (store alias or file path).
        #[arg(long)] draft_model: Option<String>,
        /// Max draft tokens (pairs with --draft-model).
        #[arg(long)] draft_max: Option<u32>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        prompt: Vec<String>,
    },
    /// Download a model (HF GGUF, URL, or registry:).
    Pull {
        model: String,
        /// Skip streaming SHA256 verification during download (size check still enforced).
        #[arg(long)]
        skip_verify: bool,
    },
    /// Upload a local model to Hugging Face (`push MODEL owner/repo[/file.gguf]`).
    Push {
        model: String,
        /// Destination `owner/repo` with optional `/file.gguf` rename.
        destination: String,
        /// Allow plain-HTTP registries (HF requires HTTPS; accepted for compat).
        #[arg(long)]
        insecure: bool,
    },
    /// List local models.
    List {},
    /// Show resident (loaded) models.
    Ps {},
    /// Stop a model: unload it from the daemon immediately.
    Stop { model: String },
    /// Show model details.
    Show {
        model: String,
        /// Print only the Modelfile (FROM + parameters + template).
        #[arg(long)] modelfile: bool,
        /// Print only the system prompt.
        #[arg(long)] system: bool,
        /// Print only the prompt template.
        #[arg(long)] template: bool,
        /// Print only the license.
        #[arg(long)] license: bool,
    },
    /// Copy/tag a model.
    Cp { source: String, dest: String },
    /// Delete one or more models (also stops them if loaded).
    Rm { models: Vec<String> },
    /// Create a model from a Modelfile.
    Create {
        model: String,
        #[arg(long, short = 'f')] file: PathBuf,
        /// Quantize after import (e.g. Q4_K_M, Q4_0, Q8_0; needs llama-quantize).
        #[arg(long, short = 'q')] quantize: Option<String>,
    },
    /// Quick throughput smoke test (no weights needed).
    Bench {
        #[arg(long, default_value = "bench-model")] model: String,
        #[arg(long, default_value_t = 128)] tokens: usize,
    },
    /// Generate shell completions (bash|zsh|fish|powershell|elvish).
    Completions { shell: clap_complete::Shell },
}
