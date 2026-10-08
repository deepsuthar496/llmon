//! Inference engine.
//!
//! Design (measured against Ollama 0.40, which ships the same llama.cpp build):
//! - One `llama-server` child per **GGUF blob** (not per user-typed alias, so
//!   `qwen` and `Qwen/...gguf` share one process), spawned lazily, reused,
//!   reaped after `LLMON_KEEP_ALIVE` seconds idle (default 300 like Ollama).
//! - **No per-model job queue.** Every request streams straight from the
//!   child's HTTP socket into the client response. The child runs
//!   `-np LLMON_NUM_PARALLEL` (default 4) slots with a unified KV pool and
//!   continuous batching, so concurrent requests decode together instead of
//!   queueing (Ollama defaults to 1 slot). Dropping the client stream drops
//!   the upstream connection, which cancels the slot immediately.
//! - Chat goes through `/v1/chat/completions`, so the backend applies the
//!   model's own Jinja chat template from GGUF metadata — the same path
//!   Ollama's llama-server runner uses — giving identical prompts/quality.
//! - Children get `PR_SET_PDEATHSIG(SIGKILL)` so they never outlive llmon.
//!
//! Backends:
//! - `llama-server` (llama.cpp) when found: real inference.
//! - Built-in stub otherwise: API-correct demo text, clearly marked.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

// ---------------------------------------------------------------- types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// Ollama-style base64 images (raw bytes, no data: prefix).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), images: Vec::new() }
    }
}

/// One generation request.
#[derive(Debug, Clone, Default)]
pub struct GenParams {
    /// Chat messages; the backend renders them with the model's template.
    pub messages: Vec<ChatMessage>,
    /// When set, sent verbatim to `/completion` (no template) — Ollama `raw`.
    pub raw_prompt: Option<String>,
    /// llama-server native sampling keys (`temperature`, `top_k`, `n_predict`,
    /// `stop`, `seed`, ...). Missing keys get Ollama's defaults.
    pub sampling: Map<String, Value>,
    /// OpenAI-style `response_format` (JSON mode / schema).
    pub response_format: Option<Value>,
    /// Thinking toggle for reasoning models (`chat_template_kwargs`).
    pub think: Option<bool>,
}

impl GenParams {
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self { messages, ..Default::default() }
    }
    pub fn raw(prompt: impl Into<String>) -> Self {
        Self { raw_prompt: Some(prompt.into()), ..Default::default() }
    }
    pub fn max_tokens(mut self, n: i64) -> Self {
        self.sampling.insert("n_predict".into(), json!(n));
        self
    }
    /// Flattened prompt text (stub backend only).
    fn flat_prompt(&self) -> String {
        if let Some(p) = &self.raw_prompt {
            return p.clone();
        }
        self.messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
    }
}

/// Generation timings, mapped 1:1 onto Ollama's final-chunk fields.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub prompt_tokens: u64,
    pub prompt_ms: f64,
    pub eval_tokens: u64,
    pub eval_ms: f64,
    pub load_ms: f64,
    pub total_ms: f64,
    pub done_reason: String,
}

#[derive(Debug, Clone)]
pub enum Event {
    Token(String),
    Thinking(String),
    Done(Stats),
    Error(String),
}

pub type EventStream = Pin<Box<dyn Stream<Item = Event> + Send + 'static>>;

/// Info for `/api/ps`.
#[derive(Debug, Clone, Serialize)]
pub struct LoadedModel {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub embedding: bool,
    pub active_requests: usize,
    pub expires_in_secs: Option<u64>,
}

// ---------------------------------------------------------------- env knobs

fn env(k: &str) -> Option<String> {
    std::env::var(format!("{}_{k}", crate::branding::ENV_PREFIX))
        .ok()
        .filter(|v| !v.trim().is_empty())
}
fn env_parse<T: std::str::FromStr>(k: &str) -> Option<T> {
    env(k).and_then(|v| v.trim().parse().ok())
}

/// Decode threads. Default = physical cores minus one on >4-core hosts and
/// all physical cores on small hosts — token generation is memory-bound and
/// Override with `LLMON_THREADS`.
pub fn default_threads() -> u32 {
    if let Some(t) = env_parse::<u32>("THREADS") {
        return t.max(1);
    }
    // Half the logical CPUs (min 1).
    // Full-width threadpools collapse on shared/oversubscribed vCPUs
    // (measured 1.3 tok/s at 4 threads vs 22.2 tok/s at 2 threads on this box).
    std::thread::available_parallelism()
        .map(|n| (n.get() as u32 / 2).max(1))
        .unwrap_or(2)
}

/// Prompt-processing (prefill) threads. Defaults to generation threads to avoid
/// threadpool contention on CPU. Override with `LLMON_THREADS_BATCH`.
pub fn default_threads_batch() -> u32 {
    env_parse::<u32>("THREADS_BATCH").unwrap_or_else(default_threads)
}

#[allow(dead_code)]
fn physical_cores() -> u32 {
    let logical = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(2);
    #[cfg(target_os = "linux")]
    {
        // Count unique (physical id, core id) pairs; respects SMT.
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut set = std::collections::HashSet::new();
            let (mut phys, mut core) = (String::new(), String::new());
            for line in s.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    match k.trim() {
                        "physical id" => phys = v.trim().to_string(),
                        "core id" => core = v.trim().to_string(),
                        _ => {}
                    }
                } else if line.trim().is_empty() && !core.is_empty() {
                    set.insert((phys.clone(), core.clone()));
                    core.clear();
                }
            }
            if !core.is_empty() {
                set.insert((phys, core));
            }
            let n = set.len() as u32;
            if n > 0 {
                // cgroup/affinity may restrict below the physical count.
                return n.min(logical);
            }
        }
    }
    logical
}

fn num_parallel() -> u32 {
    env_parse::<u32>("NUM_PARALLEL").unwrap_or(4).max(1)
}
/// Parse a keep-alive duration (`300`, `5m`, `1h`, `0`, `-1`) to seconds.
/// Mirrors the server-side parser; `None` = absent/invalid (env default).
fn parse_keep_alive_secs(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(n);
    }
    let cut = s.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(s.len());
    let n: i64 = s[..cut].parse().ok()?;
    let mult = match s[cut..].trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => 1,
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3600,
        _ => return None,
    };
    Some(n.saturating_mul(mult))
}
fn keep_alive() -> Option<Duration> {
    // Negative = forever (Ollama semantics), 0 = unload right after use.
    match env_parse::<i64>("KEEP_ALIVE") {
        Some(s) if s < 0 => None,
        Some(s) => Some(Duration::from_secs(s as u64)),
        None => Some(Duration::from_secs(300)),
    }
}
fn load_timeout() -> Duration {
    Duration::from_secs(env_parse("LOAD_TIMEOUT").unwrap_or(600))
}

/// Flags passed to every spawned `llama-server`.
/// Optimized for maximum CPU generation throughput and lowest latency:
/// - --load-mode none: loads weights directly into RAM, eliminating Linux container page faults (measured ~20% faster).
/// - -c 2048: matches Ollama default context, minimizing KV cache memory footprint.
/// - -ub 128: fine physical batch size for low decode latency on CPU.
/// - --poll 100: minimizes worker thread wake jitter.
/// - -np 4 -kvu: unified continuous batching for multi-request scaling.
/// - --spec-type ngram-mod: speculative decoding for 3x+ faster token generation.
/// - -ub 512: full prompt micro-batch processing.
pub fn backend_extra_args(
    embedding: bool,
    draft: Option<&Path>,
    manifest: Option<&crate::store::LocalManifest>,
) -> Vec<String> {
    // Precedence: explicit LLMON_* env > Modelfile PARAMETER > default.
    let pick = |env_key: &str, param_key: &str, default: String| -> String {
        if let Some(v) = env(env_key) {
            return v;
        }
        if let Some(m) = manifest {
            if let Some(v) = m.parameters.get(param_key) {
                return v.clone();
            }
        }
        default
    };
    let ctx = pick("CTX", "num_ctx", "2048".into())
        .parse::<u32>()
        .or_else(|_| env_parse::<u32>("NUM_CTX").ok_or(()))
        .unwrap_or(2048);
    let np = num_parallel();
    let mut a: Vec<String> = vec![
        "-t".into(), pick("THREADS", "num_thread", default_threads().to_string()),
        "-tb".into(), default_threads_batch().to_string(),
        "-np".into(), np.to_string(),
    ];
    // GPU layers: only passed when explicitly set (env wins, then Modelfile
    // PARAMETER num_gpu). Unset = backend default (full offload when a GPU
    // backend exists, CPU-only otherwise).
    if let Some(ngl) = env("GPU_LAYERS").or_else(|| {
        manifest.and_then(|m| m.parameters.get("num_gpu").cloned())
    }) {
        a.extend(["-ngl".into(), ngl]);
    }
    a.extend([
        "-fa".into(), env("FLASH_ATTN").unwrap_or_else(|| "auto".into()),
        "-ctk".into(), env("CACHE_TYPE_K").unwrap_or_else(|| "f16".into()),
        "-ctv".into(), env("CACHE_TYPE_V").unwrap_or_else(|| "f16".into()),
        "-kvu".into(),
        "--load-mode".into(), env("LOAD_MODE").unwrap_or_else(|| "none".into()),
        "--context-shift".into(),
        "--keep".into(), "4".into(),
    ]);
    if let Some(poll) = env("POLL") {
        a.extend(["--poll".into(), poll]);
    }
    if env("KV_PER_SLOT").as_deref() == Some("1") {
        // Guaranteed `ctx` per slot (pool = ctx * np, more RAM).
        a.extend(["--kv-unified-per-slot".into(), ctx.to_string()]);
    } else {
        // Shared pool of `ctx` tokens: a lone request gets the full window
        // (same memory as Ollama's single slot), concurrent ones share it.
        a.extend(["-c".into(), ctx.to_string()]);
    }
    if embedding {
        // Non-causal embedders need the whole input in one ubatch.
        a.extend(["--embedding".into(), "-b".into(), ctx.to_string(), "-ub".into(), ctx.to_string()]);
        let pooling = env("POOLING").unwrap_or_else(|| "mean".into());
        a.extend(["--pooling".into(), pooling]);
    } else {
        a.extend([
            "-b".into(), pick("BATCH", "num_batch", "512".into()),
            "-ub".into(), pick("UBATCH", "num_ubatch", "512".into()),
        ]);
        let spec_type = env("SPEC_TYPE").unwrap_or_else(|| "ngram-mod".into());
        if spec_type != "none" && !spec_type.is_empty() {
            a.extend(["--spec-type".into(), spec_type]);
            if let Some(m) = env("SPEC_NGRAM_MATCH") {
                a.extend(["--spec-ngram-mod-n-match".into(), m]);
            }
            if let Some(min) = env("SPEC_NGRAM_MIN") {
                a.extend(["--spec-ngram-mod-n-min".into(), min]);
            }
            if let Some(max) = env("SPEC_NGRAM_MAX") {
                a.extend(["--spec-ngram-mod-n-max".into(), max]);
            }
        }
        if let Some(d) = draft {
            a.extend(["--model-draft".into(), d.to_string_lossy().into_owned()]);
        }
        // Modelfile DRAFT_MAX wins unless the operator overrode it via env.
        let draft_max = env("SPEC_DRAFT_MAX").or_else(|| {
            manifest.and_then(|m| m.draft_max.map(|n| n.to_string()))
        });
        if let Some(n) = draft_max {
            a.extend(["--spec-draft-n-max".into(), n]);
        }
    }
    if env("MLOCK").as_deref() == Some("1") {
        a.push("--mlock".into());
    }
    // Free-form escape hatch, appended last so it wins.
    if let Some(extra) = env("BACKEND_ARGS") {
        a.extend(extra.split_whitespace().map(String::from));
    }
    a
}

/// Ollama `api.DefaultOptions()` sampling, so identical requests produce
/// identical text on llmon and Ollama.
fn apply_default_sampling(s: &mut Map<String, Value>) {
    let defaults = [
        ("temperature", json!(0.8)),
        ("top_k", json!(40)),
        ("top_p", json!(0.9)),
        ("min_p", json!(0.0)),
        ("typical_p", json!(1.0)),
        ("repeat_last_n", json!(64)),
        ("repeat_penalty", json!(1.0)),
        ("presence_penalty", json!(0.0)),
        ("frequency_penalty", json!(0.0)),
        ("n_predict", json!(-1)),
    ];
    for (k, v) in defaults {
        s.entry(k.to_string()).or_insert(v);
    }
}

// ---------------------------------------------------------------- engine

type Key = (PathBuf, bool);
type Slot = Arc<Mutex<Option<Arc<Runner>>>>;

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    slots: std::sync::Mutex<HashMap<Key, Slot>>,
    server_bin: Option<String>,
    models_dir: Option<PathBuf>,
    reaper_started: AtomicBool,
    /// `keep_alive` (secs) requested before the runner existed; applied at spawn.
    pending_keep_alive: std::sync::Mutex<HashMap<PathBuf, i64>>,
}

/// Shared HTTP client for all child traffic: one connection pool, no proxy
/// (corporate `HTTP_PROXY` must never intercept 127.0.0.1), Nagle off.
fn http() -> &'static reqwest::Client {
    static C: OnceLock<reqwest::Client> = OnceLock::new();
    C.get_or_init(|| {
        reqwest::Client::builder()
            .no_proxy()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("http client")
    })
}

/// Locate a llama.cpp helper binary: explicit env override, then system
/// multi-arch optimized builds, then PATH, then `~/.local/bin`.
pub fn find_helper(env_var: &str, bin_name: &str) -> Option<String> {
    if let Ok(p) = std::env::var(env_var) {
        if Path::new(&p).is_file() {
            return Some(p);
        }
    }
    // Explicit local installs win; Ollama's bundled copy is a last resort
    // (same llama.cpp build, but surprising when both exist).
    if let Some(p) = which(bin_name) {
        return Some(p);
    }
    let p = dirs::home_dir()?.join(".local").join("bin").join(bin_name);
    if p.is_file() {
        return Some(p.to_string_lossy().to_string());
    }
    let sys_ollama = Path::new("/usr/local/lib/ollama").join(bin_name);
    if sys_ollama.is_file() {
        return Some(sys_ollama.to_string_lossy().to_string());
    }
    None
}

fn which(name: &str) -> Option<String> {
    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let p = Path::new(dir).join(name);
        if p.is_file() {
            return Some(p.to_string_lossy().to_string());
        }
    }
    None
}

impl Engine {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(EngineInner {
                slots: std::sync::Mutex::new(HashMap::new()),
                server_bin: find_helper("LLMON_LLAMA_SERVER", "llama-server"),
                models_dir: None,
                reaper_started: AtomicBool::new(false),
                pending_keep_alive: std::sync::Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Attach the model store so the engine can resolve aliases to GGUF paths.
    pub fn with_models_dir(mut self, dir: PathBuf) -> Self {
        if let Some(i) = Arc::get_mut(&mut self.inner) {
            i.models_dir = Some(dir);
        }
        self
    }

    pub fn backend_name(&self) -> &'static str {
        if self.is_real() { "llama-server" } else { "builtin-stub" }
    }

    pub fn is_real(&self) -> bool {
        self.inner.server_bin.is_some()
    }

    fn store(&self) -> Option<crate::store::Store> {
        self.inner.models_dir.clone().map(crate::store::Store::new)
    }

    /// Resolve a user-supplied model name to its GGUF blob (exact or fuzzy).
    pub fn gguf_for(&self, model: &str) -> Option<PathBuf> {
        self.store().and_then(|s| s.gguf_path(model).ok())
    }

    fn snapshot(&self) -> Vec<Slot> {
        self.inner.slots.lock().unwrap().values().cloned().collect()
    }

    /// Names of resident models.
    #[allow(dead_code)]
    pub async fn loaded_models(&self) -> Vec<String> {
        self.loaded().await.into_iter().map(|m| m.name).collect()
    }

    /// Fast non-blocking check if a model is already resident in memory.
    pub fn has_loaded(&self, model: &str) -> bool {
        let Ok(slots) = self.inner.slots.try_lock() else { return false };
        slots.values().any(|s| {
            if let Ok(g) = s.try_lock() {
                if let Some(r) = g.as_ref() {
                    return r.name == model || r.name.starts_with(model);
                }
            }
            false
        })
    }

    pub async fn loaded(&self) -> Vec<LoadedModel> {
        let mut out = vec![];
        for slot in self.snapshot() {
            // try_lock: a slot that is mid-spawn isn't "loaded" yet.
            if let Ok(g) = slot.try_lock() {
                if let Some(r) = g.as_ref() {
                    let idle = r.idle_for();
                    out.push(LoadedModel {
                        name: r.name.clone(),
                        path: r.gguf.to_string_lossy().into_owned(),
                        size: r.size,
                        embedding: r.embedding,
                        active_requests: r.active.load(Ordering::Relaxed),
                        expires_in_secs: r.effective_keep_alive().map(|k| k.saturating_sub(idle).as_secs()),
                    });
                }
            }
        }
        out
    }

    /// Apply a per-request `keep_alive` to the runner serving `model`.
    /// `0` unloads immediately; callers should treat that as `unload`.
    /// If the runner isn't spawned yet, the value is stored and applied at
    /// spawn time (so the first request's keep-alive is honored too).
    pub async fn set_keep_alive(&self, model: &str, secs: Option<i64>) {
        let Some(gguf) = self.gguf_for(model) else { return };
        let Some(n) = secs else {
            // Unset: clear any pending override, fall back to env default.
            self.inner.pending_keep_alive.lock().unwrap().remove(&gguf);
            for slot in self.snapshot() {
                if let Ok(g) = slot.try_lock() {
                    if let Some(r) = g.as_ref().filter(|r| r.gguf == gguf) {
                        r.set_keep_alive(None);
                    }
                }
            }
            return;
        };
        let mut applied = false;
        for slot in self.snapshot() {
            if let Ok(g) = slot.try_lock() {
                if let Some(r) = g.as_ref().filter(|r| r.gguf == gguf) {
                    r.set_keep_alive(Some(n));
                    applied = true;
                }
            }
        }
        if !applied {
            self.inner.pending_keep_alive.lock().unwrap().insert(gguf, n);
        }
    }

    /// Get (or spawn) the runner for `model`. Concurrent callers for the
    /// same blob wait on one spawn. Returns load time when we spawned.
    async fn runner(&self, model: &str, embedding: bool) -> anyhow::Result<(Arc<Runner>, f64)> {
        let bin = self
            .inner
            .server_bin
            .clone()
            .ok_or_else(|| anyhow::anyhow!("llama-server not found"))?;
        let store = self.store().ok_or_else(|| anyhow::anyhow!("no model store"))?;
        let manifest = store.resolve(model)?;
        let gguf = store.blob_path(&manifest.blob_digest);
        let key = (gguf.clone(), embedding);
        let slot = self.inner.slots.lock().unwrap().entry(key).or_default().clone();
        let mut g = slot.lock().await;
        if let Some(r) = g.as_ref() {
            if r.is_alive() {
                r.touch();
                return Ok((r.clone(), 0.0));
            }
        }
        *g = None;
        // Draft model: explicit env wins, else the model's Modelfile DRAFT,
        // resolved against the store or as a filesystem path.
        let draft = if embedding {
            None
        } else {
            let spec = env("DRAFT_MODEL").or_else(|| manifest.draft.clone());
            spec.and_then(|d| {
                store.gguf_path(&d).ok().or_else(|| {
                    let p = PathBuf::from(&d);
                    p.is_file().then_some(p)
                })
            })
        };
        let adapter = if embedding {
            None
        } else {
            manifest.adapter.as_ref().and_then(|a| {
                store.gguf_path(a).ok().or_else(|| {
                    let p = PathBuf::from(a);
                    p.is_file().then_some(p)
                })
            })
        };
        // Vision projector: explicit Modelfile MMPROJ wins, else auto-detect
        // a same-repo mmproj blob (Ollama bundles it; we pair it at spawn).
        let mmproj = if embedding {
            None
        } else {
            manifest
                .mmproj
                .as_ref()
                .and_then(|x| {
                    store.gguf_path(x).ok().or_else(|| {
                        let p = PathBuf::from(x);
                        p.is_file().then_some(p)
                    })
                })
                .or_else(|| store.find_mmproj(&manifest))
        };
        let t0 = Instant::now();
        let name = format!("{}:{}", manifest.name, manifest.tag);
        let r = Arc::new(Runner::spawn(&bin, &name, &gguf, embedding, draft.as_deref(), adapter.as_deref(), mmproj.as_deref(), Some(&manifest)).await?);
        // Keep-alive precedence: per-request pending > Modelfile > env default.
        if let Some(n) = self.inner.pending_keep_alive.lock().unwrap().get(&gguf) {
            r.set_keep_alive(Some(*n));
        } else if let Some(s) = manifest.keep_alive.as_deref().and_then(parse_keep_alive_secs) {
            r.set_keep_alive(Some(s));
        }
        *g = Some(r.clone());
        drop(g);
        self.start_reaper();
        if !embedding {
            let _ = std::fs::write(store.blobs_dir().join("..").join(".last_used"), &name);
        }
        Ok((r, t0.elapsed().as_secs_f64() * 1e3))
    }

    /// Load a model without generating (Ollama: empty prompt = load).
    pub async fn load(&self, model: &str) -> anyhow::Result<f64> {
        if !self.is_real() {
            return Ok(0.0);
        }
        Ok(self.runner(model, false).await?.1)
    }

    /// Unload all runners for `model` (Ollama: `keep_alive: 0`).
    pub async fn unload(&self, model: &str) -> bool {
        let Some(gguf) = self.gguf_for(model) else { return false };
        let slots: Vec<Slot> = self
            .inner
            .slots
            .lock()
            .unwrap()
            .iter()
            .filter(|(k, _)| k.0 == gguf)
            .map(|(_, v)| v.clone())
            .collect();
        let mut any = false;
        for s in slots {
            if let Some(r) = s.lock().await.take() {
                r.kill();
                any = true;
            }
        }
        any
    }

    /// Preload models in the background (serve startup). Uses
    /// `LLMON_PRELOAD=a,b` or the last used model; `none` disables.
    pub fn preload_in_background(&self) {
        if !self.is_real() {
            return;
        }
        let list: Vec<String> = match env("PRELOAD") {
            Some(v) if v.eq_ignore_ascii_case("none") || v == "0" => vec![],
            Some(v) => v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            None => self
                .store()
                .and_then(|s| std::fs::read_to_string(s.blobs_dir().join("..").join(".last_used")).ok())
                .map(|s| vec![s.trim().to_string()])
                .unwrap_or_default(),
        };
        for m in list {
            let me = self.clone();
            tokio::spawn(async move {
                match me.load(&m).await {
                    Ok(ms) => eprintln!("{}: preloaded '{m}' in {ms:.0} ms", crate::branding::APP_BIN),
                    Err(e) => eprintln!("{}: preload '{m}' failed: {e:#}", crate::branding::APP_BIN),
                }
            });
        }
    }

    fn start_reaper(&self) {
        if self.inner.reaper_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let me = self.clone();
        tokio::spawn(async move {
            loop {
                // Tick fast enough for short keep-alives; per-runner value
                // decides the actual deadline.
                tokio::time::sleep(Duration::from_secs(1)).await;
                for slot in me.snapshot() {
                    let Ok(mut g) = slot.try_lock() else { continue };
                    let reap = match g.as_ref() {
                        Some(r) => match r.effective_keep_alive() {
                            Some(ka) => {
                                !r.is_alive()
                                    || (r.active.load(Ordering::SeqCst) == 0 && r.idle_for() >= ka)
                            }
                            None => !r.is_alive(), // forever
                        },
                        None => false,
                    };
                    if reap {
                        if let Some(r) = g.take() {
                            r.kill();
                        }
                    }
                }
            }
        });
    }

    /// Stream a generation. Dropping the stream cancels the backend slot.
    pub fn generate(&self, model: String, mut params: GenParams) -> EventStream {
        apply_default_sampling(&mut params.sampling);
        if !self.is_real() {
            return stub_stream(params);
        }
        let me = self.clone();
        Box::pin(stream! {
            let t0 = Instant::now();
            let (runner, load_ms) = match me.runner(&model, false).await {
                Ok(v) => v,
                Err(e) => { yield Event::Error(format!("{e:#}")); return; }
            };
            let _guard = runner.enter();
            let (path, body) = build_body(&params);
            let resp = http().post(format!("{}{}", runner.base, path)).json(&body).send().await;
            let resp = match resp {
                Ok(r) if r.status().is_success() => r,
                Ok(r) => {
                    let code = r.status();
                    let txt = r.text().await.unwrap_or_default();
                    yield Event::Error(format!("backend {code}: {}", backend_error_message(&txt)));
                    return;
                }
                Err(e) => {
                    runner.mark_dead_if_exited();
                    yield Event::Error(format!("backend request failed: {e}"));
                    return;
                }
            };
            let chat = params.raw_prompt.is_none();
            let mut stats = Stats { load_ms, ..Default::default() };
            let mut sse = SseLines::default();
            let mut body = resp.bytes_stream();
            'outer: while let Some(chunk) = body.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => { yield Event::Error(format!("backend stream error: {e}")); return; }
                };
                for data in sse.push(&chunk) {
                    if data == "[DONE]" { break 'outer; }
                    let Ok(v) = serde_json::from_str::<Value>(&data) else { continue };
                    if let Some(err) = v.get("error") {
                        yield Event::Error(err.get("message").and_then(Value::as_str).map(String::from).unwrap_or_else(|| err.to_string()));
                        return;
                    }
                    if let Some(t) = v.get("timings") { read_timings(t, &mut stats); }
                    if chat {
                        if let Some(c) = v.pointer("/choices/0") {
                            if let Some(s) = c.pointer("/delta/reasoning_content").and_then(Value::as_str) {
                                if !s.is_empty() { yield Event::Thinking(s.to_string()); }
                            }
                            if let Some(s) = c.pointer("/delta/content").and_then(Value::as_str) {
                                if !s.is_empty() { yield Event::Token(s.to_string()); }
                            }
                            if let Some(fr) = c.get("finish_reason").and_then(Value::as_str) {
                                stats.done_reason = fr.to_string();
                            }
                        }
                    } else {
                        if let Some(s) = v.get("content").and_then(Value::as_str) {
                            if !s.is_empty() { yield Event::Token(s.to_string()); }
                        }
                        if v.get("stop").and_then(Value::as_bool).unwrap_or(false) {
                            stats.done_reason = match v.get("stop_type").and_then(Value::as_str) {
                                Some("limit") => "length".into(),
                                _ => "stop".into(),
                            };
                            break 'outer;
                        }
                    }
                }
            }
            if stats.done_reason.is_empty() { stats.done_reason = "stop".into(); }
            stats.total_ms = t0.elapsed().as_secs_f64() * 1e3;
            yield Event::Done(stats);
        })
    }

    /// Raw passthrough to a backend endpoint (OpenAI `/v1/*`): zero
    /// re-encoding, full feature parity (tools, logprobs, response_format).
    /// The guard keeps the runner marked busy until the body is consumed.
    pub async fn proxy(
        &self,
        model: &str,
        path: &str,
        body: &Value,
        embedding: bool,
    ) -> anyhow::Result<(reqwest::Response, ActiveGuard)> {
        let (runner, _) = self.runner(model, embedding).await?;
        let guard = runner.enter();
        let resp = http()
            .post(format!("{}{}", runner.base, path))
            .json(body)
            .send()
            .await
            .map_err(|e| {
                runner.mark_dead_if_exited();
                anyhow::anyhow!("backend request failed: {e}")
            })?;
        Ok((resp, guard))
    }

    /// GET variant (for GET-only backend paths like `/props`).
    pub async fn proxy_get(
        &self,
        model: &str,
        path: &str,
        embedding: bool,
    ) -> anyhow::Result<(reqwest::Response, ActiveGuard)> {
        let (runner, _) = self.runner(model, embedding).await?;
        let guard = runner.enter();
        let resp = http()
            .get(format!("{}{}", runner.base, path))
            .send()
            .await
            .map_err(|e| {
                runner.mark_dead_if_exited();
                anyhow::anyhow!("backend request failed: {e}")
            })?;
        Ok((resp, guard))
    }

    /// Raw-bytes variant (for multipart paths like `/v1/audio/*`).
    pub async fn proxy_bytes(
        &self,
        model: &str,
        path: &str,
        content_type: String,
        body: bytes::Bytes,
    ) -> anyhow::Result<(reqwest::Response, ActiveGuard)> {
        let (runner, _) = self.runner(model, false).await?;
        let guard = runner.enter();
        let resp = http()
            .post(format!("{}{}", runner.base, path))
            .header(reqwest::header::CONTENT_TYPE, content_type)
            .body(body)
            .send()
            .await
            .map_err(|e| {
                runner.mark_dead_if_exited();
                anyhow::anyhow!("backend request failed: {e}")
            })?;
        Ok((resp, guard))
    }

    /// Real embeddings via a dedicated `--embedding` runner.
    /// Returns (vectors, prompt_tokens, load_ms).
    pub async fn embed(&self, model: &str, inputs: Vec<String>) -> anyhow::Result<(Vec<Vec<f32>>, u64, f64)> {
        if !self.is_real() {
            return Ok((inputs.iter().map(|s| pseudo_embedding(s)).collect(), 0, 0.0));
        }
        let (runner, load_ms) = self.runner(model, true).await?;
        let _g = runner.enter();
        let resp = http()
            .post(format!("{}/v1/embeddings", runner.base))
            .json(&json!({ "input": inputs, "encoding_format": "float" }))
            .send()
            .await?;
        let code = resp.status();
        let v: Value = resp.json().await?;
        if !code.is_success() {
            anyhow::bail!("backend {code}: {}", backend_error_message(&v.to_string()));
        }
        let mut data: Vec<(u64, Vec<f32>)> = v
            .get("data")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|d| {
                        let idx = d.get("index").and_then(Value::as_u64).unwrap_or(0);
                        let e = d
                            .get("embedding")
                            .and_then(Value::as_array)
                            .map(|x| x.iter().filter_map(|f| f.as_f64()).map(|f| f as f32).collect())
                            .unwrap_or_default();
                        (idx, e)
                    })
                    .collect()
            })
            .unwrap_or_default();
        data.sort_by_key(|d| d.0);
        let ptok = v.pointer("/usage/prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
        Ok((data.into_iter().map(|d| d.1).collect(), ptok, load_ms))
    }

    /// Kill every child (used on shutdown paths).
    pub async fn shutdown(&self) {
        for s in self.snapshot() {
            if let Some(r) = s.lock().await.take() {
                r.kill();
            }
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

fn build_body(p: &GenParams) -> (&'static str, Value) {
    let mut body = Map::new();
    for (k, v) in &p.sampling {
        body.insert(k.clone(), v.clone());
    }
    body.insert("stream".into(), json!(true));
    body.insert("cache_prompt".into(), json!(true));
    if let Some(prompt) = &p.raw_prompt {
        body.insert("prompt".into(), json!(prompt));
        if let Some(schema) = p.response_format.as_ref().and_then(|rf| {
            rf.pointer("/json_schema/schema").cloned().or_else(|| {
                (rf.get("type").and_then(Value::as_str) == Some("json_object")).then(|| json!({}))
            })
        }) {
            body.insert("json_schema".into(), schema);
        }
        return ("/completion", Value::Object(body));
    }
    body.insert("messages".into(), Value::Array(p.messages.iter().map(message_json).collect()));
    if let Some(rf) = &p.response_format {
        body.insert("response_format".into(), rf.clone());
    }
    if let Some(t) = p.think {
        body.insert("chat_template_kwargs".into(), json!({ "enable_thinking": t }));
    }
    ("/v1/chat/completions", Value::Object(body))
}

/// Serialize a chat message for the backend: plain `{"role","content"}`
/// normally, OpenAI multipart content when Ollama-style `images` are present.
fn message_json(m: &ChatMessage) -> Value {
    if m.images.is_empty() {
        return json!({ "role": m.role, "content": m.content });
    }
    let mut parts = vec![json!({ "type": "text", "text": m.content })];
    for img in &m.images {
        parts.push(json!({
            "type": "image_url",
            "image_url": { "url": format!("data:{};base64,{img}", sniff_mime(img)) },
        }));
    }
    json!({ "role": m.role, "content": parts })
}

/// MIME guess from base64 magic bytes (PNG/JPEG/GIF/WEBP, default png).
fn sniff_mime(b64: &str) -> &'static str {
    // Decode just enough for magic bytes.
    let head: Vec<u8> = base64_head(b64);
    if head.starts_with(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if head.starts_with(b"GIF8") {
        "image/gif"
    } else if head.len() > 11 && &head[0..4] == b"RIFF" && &head[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "image/png"
    }
}

fn base64_head(b64: &str) -> Vec<u8> {
    // 16 bytes -> 24 base64 chars; ignore padding/errors, best effort.
    let s: String = b64.chars().filter(|c| !c.is_whitespace()).take(24).collect();
    let mut out = vec![];
    let mut acc = 0u32;
    let mut bits = 0;
    for c in s.chars() {
        let v = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => continue,
        };
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8 & 0xFF);
        }
    }
    out
}

fn read_timings(t: &Value, s: &mut Stats) {
    let f = |k: &str| t.get(k).and_then(Value::as_f64);
    if let Some(v) = f("prompt_n") { s.prompt_tokens = v.max(0.0) as u64; }
    if let Some(v) = f("prompt_ms") { s.prompt_ms = v; }
    if let Some(v) = f("predicted_n") { s.eval_tokens = v.max(0.0) as u64; }
    if let Some(v) = f("predicted_ms") { s.eval_ms = v; }
    // With prompt caching `prompt_n` counts only new tokens; add cached ones
    // so prompt_eval_count reflects the real prompt like Ollama reports.
    if let Some(c) = f("cache_n") { s.prompt_tokens += c.max(0.0) as u64; }
}

fn backend_error_message(txt: &str) -> String {
    serde_json::from_str::<Value>(txt)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(String::from))
        .unwrap_or_else(|| txt.chars().take(500).collect())
}

/// Incremental SSE splitter: yields the payload of each complete `data:` line.
#[derive(Default)]
struct SseLines {
    buf: Vec<u8>,
}

impl SseLines {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = vec![];
        let mut start = 0;
        while let Some(pos) = self.buf[start..].iter().position(|&b| b == b'\n') {
            let line = &self.buf[start..start + pos];
            start += pos + 1;
            let line = std::str::from_utf8(line).unwrap_or("").trim();
            if let Some(d) = line.strip_prefix("data:") {
                out.push(d.trim_start().to_string());
            } else if line.starts_with('{') {
                out.push(line.to_string()); // non-SSE JSON error bodies
            }
        }
        self.buf.drain(..start);
        out
    }
}

// ---------------------------------------------------------------- runner

struct Runner {
    base: String,
    name: String,
    gguf: PathBuf,
    size: u64,
    embedding: bool,
    child: std::sync::Mutex<Option<tokio::process::Child>>,
    born: Instant,
    last_used_ms: AtomicU64,
    active: AtomicUsize,
    dead: AtomicBool,
    /// Per-runner keep-alive override (secs) set by the most recent request's
    /// `keep_alive`; `None` = use env default. `-1`-style unload-immediately
    /// is handled by the caller via `unload`.
    keep_alive_secs: AtomicU64,
}

/// RAII "request in flight" marker: blocks idle reaping, refreshes LRU.
pub struct ActiveGuard(Arc<Runner>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.touch();
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(target_os = "linux")]
extern "C" {
    fn prctl(option: std::os::raw::c_int, ...) -> std::os::raw::c_int;
}

impl Runner {
    fn enter(self: &Arc<Self>) -> ActiveGuard {
        self.active.fetch_add(1, Ordering::SeqCst);
        self.touch();
        ActiveGuard(self.clone())
    }
    fn touch(&self) {
        self.last_used_ms.store(self.born.elapsed().as_millis() as u64, Ordering::Relaxed);
    }
    fn idle_for(&self) -> Duration {
        let now = self.born.elapsed().as_millis() as u64;
        Duration::from_millis(now.saturating_sub(self.last_used_ms.load(Ordering::Relaxed)))
    }
    /// Per-request `keep_alive` (secs; u32::MAX = forever) overriding the env default.
    fn set_keep_alive(&self, ka: Option<i64>) {
        let v = match ka {
            Some(n) if n < 0 => u64::from(u32::MAX),          // forever
            Some(n) => n as u64,
            None => 0,                                        // unset -> env default
        };
        self.keep_alive_secs.store(v, Ordering::Relaxed);
    }
    /// Effective keep-alive duration for this runner.
    fn effective_keep_alive(&self) -> Option<Duration> {
        match self.keep_alive_secs.load(Ordering::Relaxed) {
            0 => keep_alive(),                                    // env default
            v if v == u64::from(u32::MAX) => None,                // forever
            v => Some(Duration::from_secs(v)),
        }
    }
    fn is_alive(&self) -> bool {
        if self.dead.load(Ordering::Relaxed) {
            return false;
        }
        if self.active.load(Ordering::Relaxed) > 0 {
            return true;
        }
        let mut g = self.child.lock().unwrap();
        match g.as_mut().map(|c| c.try_wait()) {
            Some(Ok(None)) => true,
            _ => {
                self.dead.store(true, Ordering::SeqCst);
                false
            }
        }
    }
    fn mark_dead_if_exited(&self) {
        let _ = self.is_alive();
    }
    fn kill(&self) {
        self.dead.store(true, Ordering::SeqCst);
        if let Some(mut c) = self.child.lock().unwrap().take() {
            let _ = c.start_kill();
            // Reap in the background so no zombie lingers.
            tokio::spawn(async move {
                let _ = c.wait().await;
            });
        }
    }

    async fn spawn(
        bin: &str,
        name: &str,
        gguf: &Path,
        embedding: bool,
        draft: Option<&Path>,
        adapter: Option<&Path>,
        mmproj: Option<&Path>,
        manifest: Option<&crate::store::LocalManifest>,
    ) -> anyhow::Result<Self> {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0")?;
            l.local_addr()?.port()
        };
        let mut cmd = tokio::process::Command::new(bin);
        if let Some(parent) = Path::new(bin).parent() {
            let p_str = parent.to_string_lossy();
            let cur_ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let new_ld = if cur_ld.is_empty() {
                p_str.to_string()
            } else if !cur_ld.split(':').any(|p| p == p_str) {
                format!("{p_str}:{cur_ld}")
            } else {
                cur_ld
            };
            cmd.env("LD_LIBRARY_PATH", new_ld);
            cmd.env("GGML_BACKEND_PATH", parent);
        }
        cmd.arg("--host").arg("127.0.0.1")
            .arg("--port").arg(port.to_string())
            .arg("--no-webui")
            .arg("-m").arg(gguf)
            .args(backend_extra_args(embedding, draft, manifest));
        if let Some(ad) = adapter {
            cmd.arg("--lora").arg(ad);
        }
        if let Some(mp) = mmproj {
            cmd.arg("--mmproj").arg(mp);
        }
        cmd.kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        let debug = env("DEBUG").as_deref() == Some("1");
        cmd.stderr(if debug { std::process::Stdio::inherit() } else { std::process::Stdio::piped() });
        #[cfg(target_os = "linux")]
        unsafe {
            // Kernel kills the child if llmon dies (even on SIGKILL).
            cmd.pre_exec(|| {
                prctl(1 /* PR_SET_PDEATHSIG */, 9 as std::os::raw::c_ulong /* SIGKILL */);
                Ok(())
            });
        }
        if debug {
            eprintln!("{}: spawn {bin} {:?}", crate::branding::APP_BIN, backend_extra_args(embedding, draft, manifest));
        }
        let mut child = cmd.spawn().map_err(|e| anyhow::anyhow!("spawn llama-server failed: {e}"))?;

        // Drain stderr continuously (a full pipe would stall the child) and
        // keep the tail for error reports.
        let tail = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<String>::new()));
        if let Some(err) = child.stderr.take() {
            let tail = tail.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(err).lines();
                while let Ok(Some(l)) = lines.next_line().await {
                    let mut t = tail.lock().unwrap();
                    if t.len() >= 30 {
                        t.pop_front();
                    }
                    t.push_back(l);
                }
            });
        }
        let base = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + load_timeout();
        let mut delay = Duration::from_millis(10);
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                tokio::time::sleep(Duration::from_millis(50)).await;
                let t = tail.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
                anyhow::bail!("llama-server exited during load ({status}):\n{t}");
            }
            if let Ok(r) = http().get(format!("{base}/health")).timeout(Duration::from_secs(2)).send().await {
                if r.status().is_success() {
                    break;
                }
            }
            if Instant::now() > deadline {
                let _ = child.kill().await;
                anyhow::bail!("llama-server did not become ready within {:?}", load_timeout());
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_millis(50));
        }
        let size = std::fs::metadata(gguf).map(|m| m.len()).unwrap_or(0);
        let r = Self {
            base,
            name: name.to_string(),
            gguf: gguf.to_path_buf(),
            size,
            embedding,
            child: std::sync::Mutex::new(Some(child)),
            born: Instant::now(),
            last_used_ms: AtomicU64::new(0),
            active: AtomicUsize::new(0),
            dead: AtomicBool::new(false),
            keep_alive_secs: AtomicU64::new(0),
        };
        Ok(r)
    }
}

/// Kill `llama-server` processes orphaned by a previous llmon that was
/// SIGKILLed before PDEATHSIG existed (parent = init, serving our blobs).
pub fn reap_orphans(models_dir: &Path) {
    #[cfg(target_os = "linux")]
    {
        let needle = models_dir.join("blobs").to_string_lossy().into_owned();
        let Ok(rd) = std::fs::read_dir("/proc") else { return };
        for e in rd.flatten() {
            let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else { continue };
            let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else { continue };
            // ppid is the 2nd field after the `(comm)` group.
            let Some(after) = stat.rsplit_once(')').map(|x| x.1) else { continue };
            let ppid = after.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok());
            if ppid != Some(1) {
                continue;
            }
            let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
            if cmdline.contains("llama-server") && cmdline.contains(&needle) {
                eprintln!("{}: killing orphaned backend pid {pid}", crate::branding::APP_BIN);
                let _ = std::process::Command::new("kill").arg("-9").arg(pid.to_string()).status();
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = models_dir;
}

// ---------------------------------------------------------------- stub

fn pseudo_embedding(s: &str) -> Vec<f32> {
    let mut v = vec![0f32; 64];
    for (i, b) in s.bytes().enumerate() {
        v[i % 64] += (b as f32) / 255.0;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

/// Deterministic stub completion. Prefix marks the backend so users can tell.
fn stub_stream(params: GenParams) -> EventStream {
    Box::pin(stream! {
        let t0 = Instant::now();
        let prompt = params.flat_prompt();
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let tail = words[words.len().saturating_sub(24)..].join(" ");
        let text = format!(
            "[{} · stub] You asked: \"{}\". Point 1) direct answer. Point 2) next step. Point 3) tip: `pull` a GGUF or install llama-server for full local inference.",
            crate::branding::APP_BIN,
            tail.chars().take(220).collect::<String>()
        );
        let max_tokens = params.sampling.get("n_predict").and_then(Value::as_i64).unwrap_or(-1);
        let max_chars = if max_tokens < 0 { usize::MAX } else { (max_tokens as usize).saturating_mul(4).max(64) };
        let (mut sent, mut n) = (0usize, 0u64);
        for w in text.split_inclusive(' ') {
            if sent >= max_chars { break; }
            sent += w.len();
            n += 1;
            yield Event::Token(w.to_string());
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        yield Event::Done(Stats { eval_tokens: n, eval_ms: ms, total_ms: ms, done_reason: "stop".into(), ..Default::default() });
    })
}
