mod branding;
mod cli;
mod config;
mod engine;
mod modelfile;
mod registry;
mod repl;
mod server;
mod store;
mod template;

use clap::Parser;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // The llama.cpp helpers are thin binaries over shared libs in
    // ~/.local/lib (see install). Make sure children can find them even
    // when the user's shell never exported LD_LIBRARY_PATH: env set here
    // is inherited by every child process we spawn.
    ensure_helper_lib_path();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = cli::Cli::parse();
    let cfg = config::Config::from_env_or(cli.host.clone(), cli.port);
    let store = store::Store::new(cfg.models_dir.clone());
    let _ = store.ensure_dirs();
    let engine = engine::Engine::new().with_models_dir(cfg.models_dir.clone());

    match cli.cmd {
        None => {
            // No subcommand: print help.
            println!("{} {} — {}", branding::APP_NAME, branding::APP_VERSION, branding::APP_TAGLINE);
            println!("Usage: {} <serve|run|pull|list|ps|stop|show|cp|rm|create|bench>", branding::APP_BIN);
            println!("Run `{} --help` for details.", branding::APP_BIN);
        }
        Some(cli::Cmd::Serve { host, port, ctx_size, threads, batch_size }) => {
            // CLI perf flags win over env for this process (spawns are lazy,
            // so setting them here covers every future backend child).
            if let Some(n) = ctx_size {
                std::env::set_var("LLMON_CTX", n.to_string());
            }
            if let Some(n) = threads {
                std::env::set_var("LLMON_THREADS", n.to_string());
            }
            if let Some(n) = batch_size {
                std::env::set_var("LLMON_BATCH", n.to_string());
            }
            let cfg = config::Config::from_env_or(host.or(cli.host), port.or(cli.port));
            serve(cfg, store, engine).await?;
        }
        Some(cli::Cmd::Run { model, verbose, keepalive, format, think, hidethinking, ctx_size, threads, batch_size, draft_model, draft_max, prompt }) => {
            // Perf overrides apply to backends this command spawns. A live
            // daemon keeps its own serve flags (shared process).
            if let Some(n) = ctx_size {
                std::env::set_var("LLMON_CTX", n.to_string());
            }
            if let Some(n) = threads {
                std::env::set_var("LLMON_THREADS", n.to_string());
            }
            if let Some(n) = batch_size {
                std::env::set_var("LLMON_BATCH", n.to_string());
            }
            if let Some(d) = draft_model {
                std::env::set_var("LLMON_DRAFT_MODEL", d);
            }
            if let Some(n) = draft_max {
                std::env::set_var("LLMON_SPEC_DRAFT_MAX", n.to_string());
            }
            // Daemon if running (model resident, shared), else one
            // in-process backend kept alive for the whole session.
            if engine.is_real() && engine.gguf_for(&model).is_none() {
                eprintln!("note: model '{model}' not in store — run `pull` first");
            }
            let opts = repl::SessionOptions {
                verbose,
                format,
                keep_alive: keepalive,
                hide_thinking: hidethinking,
                think: think.map(|t| match t.to_lowercase().as_str() {
                    "false" | "0" | "off" | "no" => false,
                    _ => true,
                }),
                ..Default::default()
            };
            let backend = repl::Backend::connect(&cfg.base_url(), engine.clone()).await;
            let res = if prompt.is_empty() {
                repl::chat_loop(&backend, store, &model, verbose, opts).await
            } else {
                repl::one_shot(&backend, &model, &prompt.join(" "), &opts).await
            };
            engine.shutdown().await;
            res?;
        }
        Some(cli::Cmd::Pull { model, skip_verify }) => {
            // `pull <spec>`: spec may be `owner/repo/file.gguf`, URL, hf:, registry:.
            let skip_verify = skip_verify
                || std::env::var("LLMON_SKIP_VERIFY")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
            let verify = !skip_verify;
            let (name, _tag) = store::Store::split_name(&model);
            let (url, fname) = registry::resolve_pull_url(&name).await?;
            println!("pulling {model}\n  from {url}");
            let dest_tmp = store.blobs_dir().join(format!("{}-{}", safe(&name), safe(&fname)));
            std::fs::create_dir_all(store.blobs_dir())?;
            let (size, digest) =
                registry::download_to(&url, &dest_tmp, verify, None::<fn(u64, Option<u64>)>).await?;
            let final_path = store.blob_path(&digest);
            if dest_tmp != final_path {
                if final_path.exists() {
                    std::fs::remove_file(&dest_tmp)?;
                } else {
                    std::fs::rename(&dest_tmp, &final_path)?;
                }
            }
            let (n, t) = store::Store::split_name(&model);
            store.put_alias(
                &n,
                &t,
                &store::LocalManifest {
                    name: n.clone(),
                    tag: t.clone(),
                    registry: "hf".into(),
                    blob_digest: digest.clone(),
                    size,
                    parameter_size: None,
                    quantization: None,
                    system: None,
                    template: None,
                    license: None,
                    verified: Some(verify),
                    messages: Vec::new(),
                    draft: None,
                    adapter: None,
                    mmproj: None,
                    draft_max: None,
                    keep_alive: None,
                    parameters: Default::default(),
                    created_at: store::now_rfc3339(),
                },
            )?;
            if verify {
                println!("verified {digest} in a single pass ({size} bytes, no re-read)");
            } else {
                println!("pulled {digest} ({size} bytes, verification skipped)");
            }
            println!("pulled {n}:{t} ({size} bytes)");
        }
        Some(cli::Cmd::Push { model, destination, insecure }) => {
            // `push MODEL owner/repo[/file.gguf]`: upload the model's GGUF
            // to Hugging Face. Needs HF_TOKEN/HF_HUB_TOKEN with write access.
            if insecure {
                eprintln!("note: --insecure is accepted for OCI-registry compat but Hugging Face requires HTTPS");
            }
            let m = store.resolve(&model)?;
            let gguf = store.blob_path(&m.blob_digest);
            let (repo, mut filename) = split_push_dest(&destination)?;
            if filename.is_empty() {
                filename = format!("{}-{}.gguf", safe(&m.name), safe(&m.tag));
            }
            println!("pushing {}:{} -> {repo}/{filename}", m.name, m.tag);
            let msg = registry::push_to_hf(&repo, &gguf, &filename, None::<fn(u64, Option<u64>)>).await?;
            println!("{msg}");
        }
        Some(cli::Cmd::List {}) => {
            let models = store.list()?;
            if models.is_empty() {
                println!("no models — try `{} pull <owner/repo/file.gguf>`", branding::APP_BIN);
            }
            for m in models {
                println!("{}:{}\t{}\t{} bytes", m.name, m.tag, m.blob_digest, m.size);
            }
        }
        Some(cli::Cmd::Ps {}) => {
            // Resident models live in the daemon, not in this CLI process.
            let url = format!("{}/api/ps", cfg.base_url());
            match reqwest::Client::builder().no_proxy().build()?.get(&url).send().await {
                Ok(r) => {
                    let v: serde_json::Value = r.json().await.unwrap_or_default();
                    println!("NAME\tSIZE\tACTIVE\tEXPIRES_IN");
                    for m in v["models"].as_array().cloned().unwrap_or_default() {
                        println!(
                            "{}{}\t{} MB\t{}\t{}",
                            m["name"].as_str().unwrap_or(""),
                            if m["embedding"].as_bool() == Some(true) { " (embed)" } else { "" },
                            m["size"].as_u64().unwrap_or(0) / 1_000_000,
                            m["active_requests"],
                            m["expires_in_secs"].as_u64().map(|s| format!("{s}s")).unwrap_or("forever".into()),
                        );
                    }
                }
                Err(_) => println!("daemon not running at {}", cfg.base_url()),
            }
        }
        Some(cli::Cmd::Stop { model }) => {
            // Ask the daemon to unload the model now (Ollama: keep_alive=0).
            // Silent no-op when the daemon isn't running, like `ps`.
            let resolved = store
                .resolve(&model)
                .map(|m| format!("{}:{}", m.name, m.tag))
                .unwrap_or(model.clone());
            let url = format!("{}/api/generate", cfg.base_url());
            let body = serde_json::json!({ "model": resolved, "keep_alive": 0 });
            match reqwest::Client::builder().no_proxy().build()?.post(&url).json(&body).send().await {
                Ok(r) if r.status().is_success() => println!("stopped {model}"),
                Ok(r) => println!("couldn't stop {model}: HTTP {}", r.status()),
                Err(_) => println!("daemon not running at {}", cfg.base_url()),
            }
        }
        Some(cli::Cmd::Show { model, modelfile, system, template, license }) => {
            let m = store.resolve(&model)?;
            if modelfile {
                // Ollama `show --modelfile`: reconstruct a Modelfile.
                println!("FROM {}", m.blob_digest);
                if let Some(q) = &m.quantization {
                    println!("# quantization: {q}");
                }
                if let Some(s) = &m.system {
                    println!("SYSTEM {:?}", s);
                }
                if let Some(t) = &m.template {
                    println!("TEMPLATE {:?}", t);
                }
                if let Some(l) = &m.license {
                    println!("LICENSE {:?}", l);
                }
            } else if system {
                println!("{}", m.system.as_deref().unwrap_or(""));
            } else if template {
                println!("{}", m.template.as_deref().unwrap_or(""));
            } else if license {
                println!("{}", m.license.as_deref().unwrap_or(""));
            } else {
                println!("{}", serde_json::to_string_pretty(&m)?);
            }
        }
        Some(cli::Cmd::Cp { source, dest }) => {
            store.copy(&source, &dest)?;
            println!("copied {source} -> {dest}");
        }
        Some(cli::Cmd::Rm { models }) => {
            let mut failures = 0;
            for model in &models {
                let resolved = store
                    .resolve(model)
                    .map(|m| format!("{}:{}", m.name, m.tag))
                    .unwrap_or(model.clone());
                match store.remove(&resolved) {
                    Ok(true) => println!("deleted {model}"),
                    Ok(false) => {
                        println!("{model} not found");
                        failures += 1;
                    }
                    Err(e) => {
                        println!("couldn't delete {model}: {e}");
                        failures += 1;
                    }
                }
            }
            if failures > 0 {
                std::process::exit(1);
            }
        }
        Some(cli::Cmd::Create { model, file, quantize }) => {
            let mf = modelfile::parse(&file)?;
            let (n, t) = store::Store::split_name(&model);
            let from_p = std::path::Path::new(&mf.from);
            if from_p.exists() {
                let mut m = store.import_file(&n, &t, from_p)?;
                m.system = mf.system;
                m.template = mf.template;
                m.license = mf.license;
                m.messages = mf
                    .messages
                    .iter()
                    .map(|x| store::Message { role: x.role.clone(), content: x.content.clone() })
                    .collect();
                m.draft = mf.draft;
                m.keep_alive = mf.keep_alive;
                m.draft_max = mf.draft_max;
                m.adapter = mf.adapter;
                m.mmproj = mf.mmproj;
m.parameters = mf.parameters.clone();
                if let Some(qtype) = quantize {
                    quantize_blob(&store, &mut m, &qtype)?;
                }
                store.put_alias(&n, &t, &m)?;
                println!("created {n}:{t}");
            } else {
                anyhow::bail!("FROM target not a local file: {} (remote FROM via server /api/create)", mf.from);
            }
        }
        Some(cli::Cmd::Completions { shell }) => {
            use clap::CommandFactory;
            let mut cmd = cli::Cli::command();
            clap_complete::generate(shell, &mut cmd, branding::APP_BIN, &mut std::io::stdout());
        }
        Some(cli::Cmd::Bench { model, tokens }) => {
            let t0 = std::time::Instant::now();
            let params = engine::GenParams::raw("benchmark prompt").max_tokens(tokens as i64);
            let mut s = engine.generate(model, params);
            let (mut chunks, mut stats) = (0usize, None);
            while let Some(ev) = s.next().await {
                match ev {
                    engine::Event::Token(_) => chunks += 1,
                    engine::Event::Done(st) => stats = Some(st),
                    engine::Event::Error(e) => anyhow::bail!(e),
                    engine::Event::Thinking(_) => {}
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            let st = stats.unwrap_or_default();
            let n = if st.eval_tokens > 0 { st.eval_tokens as usize } else { chunks.max(1) };
            println!("{n} tokens in {dt:.2}s = {:.1} tok/s wall (engine: {})", n as f64 / dt, engine.backend_name());
            if st.eval_ms > 0.0 {
                println!("decode {:.1} tok/s, load {:.0} ms", st.eval_tokens as f64 / (st.eval_ms / 1e3), st.load_ms);
            }
            engine.shutdown().await;
        }
    }
    Ok(())
}

async fn serve(cfg: config::Config, store: store::Store, engine: engine::Engine) -> anyhow::Result<()> {
    let addr = cfg.addr();
    engine::reap_orphans(&cfg.models_dir);
    println!("{} {} serving on http://{} ({})", branding::APP_NAME, branding::APP_VERSION, addr, engine.backend_name());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    // Warm the last-used (or LLMON_PRELOAD) model so the first request
    // doesn't pay spawn + weight load.
    engine.preload_in_background();
    let app = server::router(cfg, store, engine.clone());
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("signal handler");
                tokio::select! { _ = ctrl_c => {}, _ = term.recv() => {} }
            }
            #[cfg(not(unix))]
            let _ = ctrl_c.await;
        })
        .await?;
    engine.shutdown().await;
    Ok(())
}

fn safe(s: &str) -> String {
    s.replace(['/', '\\', ':', ' '], "-")
}

/// Quantize a stored blob in place via `llama-quantize` and re-point the
/// manifest at the new blob (old blob is garbage-collected if unreferenced).
fn quantize_blob(store: &store::Store, m: &mut store::LocalManifest, qtype: &str) -> anyhow::Result<()> {
    let bin = engine::find_helper("LLMON_LLAMA_QUANTIZE", "llama-quantize").ok_or_else(|| {
        anyhow::anyhow!("llama-quantize not found (set LLMON_LLAMA_QUANTIZE or install the backend)")
    })?;
    let src = store.blob_path(&m.blob_digest);
    let tmp = store.blobs_dir().join(format!("quant-{}.gguf", safe(&qtype)));
    println!("quantizing to {qtype} (this reads the full model once)...");
    let st = std::process::Command::new(&bin)
        .arg("--allow-requantize")
        .arg(&src)
        .arg(&tmp)
        .arg(qtype)
        .arg(std::thread::available_parallelism().map(|n| (n.get() / 2).max(1).to_string()).unwrap_or_else(|_| "2".into()))
        .status()?;
    if !st.success() {
        anyhow::bail!("llama-quantize failed (exit {st}); is '{qtype}' a valid type for this model?");
    }
    // Hash the result into the content-addressed store.
    let digest = block_on_hash(&tmp)?;
    let size = std::fs::metadata(&tmp)?.len();
    let dst = store.blob_path(&digest);
    if tmp != dst {
        if dst.exists() {
            std::fs::remove_file(&tmp)?;
        } else {
            std::fs::rename(&tmp, &dst)?;
        }
    }
    let old = std::mem::replace(&mut m.blob_digest, digest);
    m.size = size;
    m.quantization = Some(qtype.to_uppercase());
    // GC the old blob when nothing references it anymore.
    let still_used = store.list().map(|l| l.iter().any(|x| x.blob_digest == old)).unwrap_or(true);
    if !still_used {
        let _ = std::fs::remove_file(store.blob_path(&old));
    }
    Ok(())
}

/// Synchronous sha256 of a file (small helper; streaming, bounded RAM).
fn block_on_hash(path: &std::path::Path) -> anyhow::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = sha2::Sha256::new();
    use sha2::Digest;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("sha256-{}", hex::encode(h.finalize())))
}

/// Split `owner/repo[/file.gguf]` push destination into repo + filename.
fn split_push_dest(dest: &str) -> anyhow::Result<(String, String)> {
    let dest = dest.strip_prefix("hf:").unwrap_or(dest);
    let parts: Vec<&str> = dest.split('/').collect();
    if parts.len() < 2 {
        anyhow::bail!("destination must be owner/repo[/file.gguf], got '{dest}'");
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let filename = if parts.len() > 2 {
        parts[2..].join("/")
    } else {
        String::new() // caller fills in the model's own filename below
    };
    Ok((repo, filename))
}

/// Prepend helper library directories to LD_LIBRARY_PATH (once) so spawned
/// llama-server/llama-cli children resolve their shared SIMD/backend libraries.
fn ensure_helper_lib_path() {
    let mut dirs_to_add = vec![];
    if std::path::Path::new("/usr/local/lib/ollama").is_dir() {
        dirs_to_add.push("/usr/local/lib/ollama".to_string());
    }
    if let Some(h) = dirs::home_dir() {
        let lib = h.join(".local").join("lib");
        if lib.is_dir() {
            dirs_to_add.push(lib.to_string_lossy().to_string());
        }
    }
    for lib in dirs_to_add {
        let cur = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        if !cur.split(':').any(|p| p == lib) {
            let next = if cur.is_empty() { lib } else { format!("{lib}:{cur}") };
            std::env::set_var("LD_LIBRARY_PATH", next);
        }
    }
}
