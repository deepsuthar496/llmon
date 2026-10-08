//! HTTP daemon: Ollama-native `/api/*` + OpenAI-compatible `/v1/*`.
//!
//! - `/api/generate|chat` stream NDJSON exactly like Ollama, ending with a
//!   `done: true` chunk carrying `total_duration`, `load_duration`,
//!   `prompt_eval_count/duration`, `eval_count/duration`, `done_reason`.
//! - `/v1/*` is a byte-level passthrough to the backend's OpenAI server
//!   (tools, logprobs, response_format, usage all native, zero re-encoding).
//! - Embeddings are real model embeddings via a dedicated backend runner.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::engine::{ChatMessage, Engine, Event, GenParams, Stats};
use crate::store::Store;

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    pub cfg: Arc<Config>,
    pub store: Arc<Store>,
    pub engine: Engine,
    pub started: Instant,
}

pub fn router(cfg: Config, store: Store, engine: Engine) -> Router {
    let st = AppState {
        cfg: Arc::new(cfg),
        store: Arc::new(store),
        engine,
        started: Instant::now(),
    };
    Router::new()
        // health / meta
        .route("/", get(|| async { "llmon is running" }))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/api/version", get(api_version))
        // Ollama-native
        .route("/api/tags", get(api_tags))
        .route("/api/ps", get(api_ps))
        .route("/api/show", post(api_show))
        .route("/api/generate", post(api_generate))
        .route("/api/chat", post(api_chat))
        .route("/api/embed", post(api_embed))
        .route("/api/embeddings", post(api_embeddings_legacy))
        .route("/api/pull", post(api_pull))
        .route("/api/push", post(api_push))
        .route("/api/create", post(api_create))
        .route("/api/copy", post(api_copy))
        .route("/api/delete", delete(api_delete))
        .route("/api/status", get(api_status))
        .route("/api/blobs/:digest", post(api_blob_upload).head(api_blob_exists))
        // experimental (Ollama cloud features — single-binary has no cloud)
        .route("/api/experimental/web_search", post(api_web_unsupported))
        .route("/api/experimental/web_fetch", post(api_web_unsupported))
        .route("/api/experimental/model-recommendations", get(api_model_recommendations))
        // OpenAI-compatible
        .route("/v1/models", get(v1_models))
        .route("/v1/models/:model", get(v1_model_show))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/embeddings", post(v1_embeddings))
        // tokenizer (proxy to backend, native llama.cpp)
        .route("/tokenize", post(v1_tokenize))
        .route("/detokenize", post(v1_detokenize))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/reranking", post(v1_rerank))
        .route("/rerank", post(v1_rerank))
        .route("/v1/messages", post(v1_messages))
        .route("/v1/messages/count_tokens", post(v1_messages_count_tokens))
        .route("/v1/responses", post(v1_responses))
        .route("/v1/responses/input_tokens", post(v1_responses_input_tokens))
        .route("/v1/load_lora_adapter", post(v1_load_lora))
        .route("/v1/unload_lora_adapter", post(v1_unload_lora))
        .route("/tokenizer_info", get(v1_tokenizer_info))
        .route("/v1/audio/transcriptions", post(v1_audio_transcriptions))
        .route("/v1/audio/translations", post(v1_audio_translations))
        .layer(CorsLayer::permissive())
        .with_state(st)
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok", "app": crate::branding::APP_NAME }))
}

async fn metrics(State(st): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "app": crate::branding::APP_NAME,
        "uptime_secs": st.started.elapsed().as_secs(),
        "backend": st.engine.backend_name(),
        "loaded": st.engine.loaded().await,
    }))
}

async fn api_version() -> impl IntoResponse {
    Json(json!({ "version": crate::branding::APP_VERSION }))
}

/// Ollama `/api/status`: lightweight daemon state (no model listing).
async fn api_status(State(st): State<AppState>) -> impl IntoResponse {
    let loaded = st.engine.loaded().await;
    Json(json!({
        "app": crate::branding::APP_NAME,
        "version": crate::branding::APP_VERSION,
        "status": "ok",
        "backend": st.engine.backend_name(),
        "resident_models": loaded.len(),
        "uptime_secs": st.started.elapsed().as_secs(),
    }))
}

/// `POST /api/blobs/:digest`: store raw bytes as a content-addressed blob
/// after verifying the sha256 matches `:digest` (OCI `sha256:<hex>` or
/// llmon `sha256-<hex>` form). Staging primitive for OCI-style pushes.
async fn api_blob_upload(
    State(st): State<AppState>,
    axum::extract::Path(digest): axum::extract::Path<String>,
    body: axum::body::Bytes,
) -> Response {
    use sha2::Digest;
    let hex = digest.strip_prefix("sha256:").or_else(|| digest.strip_prefix("sha256-")).unwrap_or(&digest);
    let actual = hex::encode(sha2::Sha256::digest(&body));
    if actual != hex {
        return err(
            StatusCode::BAD_REQUEST,
            format!("digest mismatch: url says {hex}, bytes hash to {actual}"),
        );
    }
    let path = st.store.blob_path(&format!("sha256-{hex}"));
    if let Some(p) = path.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    match std::fs::write(&path, &body) {
        Ok(()) => (StatusCode::CREATED, Json(json!({ "status": "stored", "digest": format!("sha256-{hex}"), "size": body.len() }))).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("write blob: {e}")),
    }
}

/// `HEAD /api/blobs/:digest`: 200 when the blob exists, else 404.
async fn api_blob_exists(
    State(st): State<AppState>,
    axum::extract::Path(digest): axum::extract::Path<String>,
) -> Response {
    let hex = digest.strip_prefix("sha256:").or_else(|| digest.strip_prefix("sha256-")).unwrap_or(&digest);
    if st.store.blob_path(&format!("sha256-{hex}")).is_file() {
        StatusCode::OK.into_response()
    } else {
        err(StatusCode::NOT_FOUND, format!("blob {digest} not found"))
    }
}

/// Ollama's web_search/web_fetch proxy the ollama.com cloud. llmon is a
/// local single binary, so these fail with an explicit, documented error
/// instead of silently returning empty results.
async fn api_web_unsupported() -> impl IntoResponse {
    err_req(
        StatusCode::BAD_REQUEST,
        "web search/fetch is only available through Ollama cloud; llmon runs fully local",
    )
}

/// Model recommendations: no telemetry, so an empty list (valid response).
async fn api_model_recommendations() -> impl IntoResponse {
    Json(json!({ "recommendations": [] }))
}

fn err_req(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({ "error": msg }))).into_response()
}

async fn api_tags(State(st): State<AppState>) -> impl IntoResponse {
    let models = st.store.list().unwrap_or_default();
    Json(json!({
        "models": models.iter().map(|m| json!({
            "name": format!("{}:{}", m.name, m.tag),
            "model": format!("{}:{}", m.name, m.tag),
            "modified_at": m.created_at,
            "size": m.size,
            "digest": m.blob_digest,
            "details": { "format": "gguf", "quantization_level": m.quantization, "parameter_size": m.parameter_size },
        })).collect::<Vec<_>>(),
    }))
}

async fn api_ps(State(st): State<AppState>) -> impl IntoResponse {
    let loaded = st.engine.loaded().await;
    Json(json!({
        "models": loaded.iter().map(|m| json!({
            "name": m.name, "model": m.name, "size": m.size,
            "embedding": m.embedding, "active_requests": m.active_requests,
            "expires_in_secs": m.expires_in_secs,
        })).collect::<Vec<_>>(),
    }))
}

#[derive(Deserialize)]
struct ShowReq {
    #[serde(alias = "name")]
    model: String,
}
async fn api_show(State(st): State<AppState>, Json(req): Json<ShowReq>) -> impl IntoResponse {
    match st.store.resolve(&req.model) {
        Ok(m) => Json(json!({
            "model": format!("{}:{}", m.name, m.tag),
            "details": { "format": "gguf", "parameter_size": m.parameter_size, "quantization_level": m.quantization },
            "model_info": { "general.file_type": m.quantization },
            "system": m.system.clone().unwrap_or_default(),
            "template": m.template.clone().unwrap_or_default(),
            "license": m.license.clone().unwrap_or_default(),
        }))
        .into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e.to_string()),
    }
}

// ---------- helpers ----------

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

/// RFC 3339 UTC timestamp (Ollama's `created_at`), no chrono dependency.
fn now_rfc3339() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60,
        d.subsec_nanos()
    )
}

/// Translate Ollama `options` into llama-server sampling keys.
fn ollama_options(opts: &Option<Map<String, Value>>) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(o) = opts else { return out };
    for (k, v) in o {
        let key = match k.as_str() {
            "num_predict" => "n_predict",
            "num_keep" => "n_keep",
            "temperature" | "top_k" | "top_p" | "min_p" | "typical_p" | "seed" | "stop"
            | "repeat_penalty" | "repeat_last_n" | "presence_penalty" | "frequency_penalty"
            | "mirostat" | "mirostat_tau" | "mirostat_eta" => k.as_str(),
            // Load-time options (num_ctx, num_thread, num_gpu, ...) are
            // daemon-level env knobs in llmon; accept and ignore.
            _ => continue,
        };
        if !v.is_null() {
            out.insert(key.to_string(), v.clone());
        }
    }
    out
}

/// Ollama `format`: "json" or a JSON schema → OpenAI `response_format`.
fn ollama_format(format: &Option<Value>) -> Option<Value> {
    match format {
        Some(Value::String(s)) if s == "json" => Some(json!({ "type": "json_object" })),
        Some(v @ Value::Object(_)) => Some(json!({ "type": "json_schema", "json_schema": { "schema": v } })),
        _ => None,
    }
}

fn ns(ms: f64) -> u64 {
    (ms * 1e6).max(0.0) as u64
}

fn final_fields(m: &mut Map<String, Value>, s: &Stats) {
    m.insert("done".into(), json!(true));
    m.insert("done_reason".into(), json!(s.done_reason));
    m.insert("total_duration".into(), json!(ns(s.total_ms)));
    m.insert("load_duration".into(), json!(ns(s.load_ms)));
    m.insert("prompt_eval_count".into(), json!(s.prompt_tokens));
    m.insert("prompt_eval_duration".into(), json!(ns(s.prompt_ms)));
    m.insert("eval_count".into(), json!(s.eval_tokens));
    m.insert("eval_duration".into(), json!(ns(s.eval_ms)));
}

fn ndjson_line(v: &Value) -> Result<bytes::Bytes, std::convert::Infallible> {
    let mut s = serde_json::to_vec(v).unwrap_or_default();
    s.push(b'\n');
    Ok(bytes::Bytes::from(s))
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Generate,
    Chat,
}

/// Shared streaming/non-streaming responder for /api/generate and /api/chat.
async fn respond(st: AppState, model: String, params: GenParams, stream_mode: bool, kind: Kind) -> Response {
    let mut events = st.engine.generate(model.clone(), params);
    let shell = move |content: &str, thinking: Option<&str>| -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("model".into(), json!(model));
        m.insert("created_at".into(), json!(now_rfc3339()));
        match kind {
            Kind::Generate => {
                m.insert("response".into(), json!(content));
                if let Some(t) = thinking {
                    m.insert("thinking".into(), json!(t));
                }
            }
            Kind::Chat => {
                let mut msg = json!({ "role": "assistant", "content": content });
                if let Some(t) = thinking {
                    msg["thinking"] = json!(t);
                }
                m.insert("message".into(), msg);
            }
        }
        m
    };

    if !stream_mode {
        let (mut text, mut think, mut stats) = (String::new(), String::new(), None);
        while let Some(ev) = events.next().await {
            match ev {
                Event::Token(t) => text.push_str(&t),
                Event::Thinking(t) => think.push_str(&t),
                Event::Done(s) => stats = Some(s),
                Event::Error(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            }
        }
        let mut m = shell(&text, (!think.is_empty()).then_some(think.as_str()));
        final_fields(&mut m, &stats.unwrap_or_default());
        return Json(Value::Object(m)).into_response();
    }

    let s = async_stream::stream! {
        while let Some(ev) = events.next().await {
            match ev {
                Event::Token(t) => {
                    let mut m = shell(&t, None);
                    m.insert("done".into(), json!(false));
                    yield ndjson_line(&Value::Object(m));
                }
                Event::Thinking(t) => {
                    let mut m = shell("", Some(&t));
                    m.insert("done".into(), json!(false));
                    yield ndjson_line(&Value::Object(m));
                }
                Event::Done(s) => {
                    let mut m = shell("", None);
                    final_fields(&mut m, &s);
                    yield ndjson_line(&Value::Object(m));
                }
                Event::Error(e) => {
                    yield ndjson_line(&json!({ "error": e }));
                    break;
                }
            }
        }
    };
    ([("content-type", "application/x-ndjson")], axum::body::Body::from_stream(s)).into_response()
}

/// Parse Ollama `keep_alive`: number (secs), duration string ("5m","1h","0"),
/// or `null`/absent (env default). Returns `(secs, is_unload, is_forever)`.
fn parse_keep_alive(v: &Option<Value>) -> Option<i64> {
    let v = v.as_ref()?;
    match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                return None;
            }
            if let Ok(n) = s.parse::<i64>() {
                return Some(n);
            }
            parse_duration(s)
        }
        Value::Null => None,
        _ => None,
    }
}

/// Duration string -> seconds ("5m" -> 300, "1h30m" -> 5400, "-1" -> -1).
fn parse_duration(s: &str) -> Option<i64> {
    let (digits, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(s.len()));
    let n: i64 = digits.parse().ok()?;
    let mult = match unit.trim() {
        "" | "s" | "sec" | "secs" | "second" | "seconds" => 1,
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3600,
        _ => return None,
    };
    Some(n.saturating_mul(mult))
}

/// Apply per-request keep-alive semantics. Returns true when the request is
/// an immediate unload (keep_alive 0), which callers handle as `unload`.
async fn apply_keep_alive(st: &AppState, model: &str, keep_alive: &Option<Value>) -> bool {
    let secs = parse_keep_alive(keep_alive);
    match secs {
        Some(0) => true,
        Some(n) => {
            st.engine.set_keep_alive(model, Some(n)).await;
            false
        }
        None => false,
    }
}

/// Ollama semantics: empty prompt/messages = load (or unload with keep_alive 0).
async fn load_only(st: &AppState, model: &str, keep_alive: &Option<Value>, kind: Kind) -> Response {
    let unload = matches!(keep_alive, Some(Value::Number(n)) if n.as_f64() == Some(0.0))
        || matches!(keep_alive, Some(Value::String(s)) if s == "0" || s == "0s");
    let mut stats = Stats::default();
    let reason = if unload {
        st.engine.unload(model).await;
        "unload"
    } else {
        match st.engine.load(model).await {
            Ok(ms) => stats.load_ms = ms,
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
        "load"
    };
    stats.total_ms = stats.load_ms;
    stats.done_reason = reason.into();
    let mut m = Map::new();
    m.insert("model".into(), json!(model));
    m.insert("created_at".into(), json!(now_rfc3339()));
    match kind {
        Kind::Generate => { m.insert("response".into(), json!("")); }
        Kind::Chat => { m.insert("message".into(), json!({ "role": "assistant", "content": "" })); }
    }
    final_fields(&mut m, &stats);
    Json(Value::Object(m)).into_response()
}

// ---------- generate / chat (Ollama-native) ----------

#[derive(Deserialize)]
struct GenerateReq {
    model: String,
    #[serde(default)]
    prompt: String,
    system: Option<String>,
    template: Option<String>,
    #[serde(default)]
    raw: bool,
    stream: Option<bool>,
    format: Option<Value>,
    think: Option<bool>,
    keep_alive: Option<Value>,
    options: Option<Map<String, Value>>,
}

async fn api_generate(State(st): State<AppState>, Json(req): Json<GenerateReq>) -> Response {
    if let Err(e) = ensure_model(&st, &req.model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    if apply_keep_alive(&st, &req.model, &req.keep_alive).await {
        return load_only(&st, &req.model, &req.keep_alive, Kind::Generate).await;
    }
    if req.prompt.is_empty() && req.system.is_none() {
        return load_only(&st, &req.model, &req.keep_alive, Kind::Generate).await;
    }
    let manifest = st.store.resolve(&req.model).ok();
    let system = req.system.clone().or_else(|| manifest.as_ref().and_then(|m| m.system.clone()));
    let template = req.template.clone().or_else(|| manifest.as_ref().and_then(|m| m.template.clone()));
    let mut params = if req.raw {
        GenParams::raw(req.prompt.clone())
    } else if let Some(tpl) = template.filter(|t| !t.trim().is_empty()) {
        // Explicit Modelfile/request template: render it ourselves.
        GenParams::raw(crate::template::apply(Some(&tpl), system.as_deref(), &req.prompt, None))
    } else {
        // Default: the model's own chat template (same as Ollama).
        let mut msgs = vec![];
        if let Some(s) = system.filter(|s| !s.is_empty()) {
            msgs.push(ChatMessage::new("system", s));
        }
        msgs.push(ChatMessage::new("user", req.prompt.clone()));
        GenParams::chat(msgs)
    };
    params.sampling = ollama_options(&req.options);
    params.response_format = ollama_format(&req.format);
    params.think = req.think;
    respond(st, req.model, params, req.stream.unwrap_or(true), Kind::Generate).await
}

#[derive(Deserialize)]
struct ChatReq {
    model: String,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    format: Option<Value>,
    think: Option<bool>,
    keep_alive: Option<Value>,
    options: Option<Map<String, Value>>,
}

async fn api_chat(State(st): State<AppState>, Json(req): Json<ChatReq>) -> Response {
    if let Err(e) = ensure_model(&st, &req.model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    if apply_keep_alive(&st, &req.model, &req.keep_alive).await {
        return load_only(&st, &req.model, &req.keep_alive, Kind::Chat).await;
    }
    if req.messages.is_empty() {
        return load_only(&st, &req.model, &req.keep_alive, Kind::Chat).await;
    }
    let mut messages = req.messages;
    let manifest = st.store.resolve(&req.model).ok();
    // Modelfile SYSTEM applies when the client didn't send one.
    if !messages.iter().any(|m| m.role == "system") {
        if let Some(sys) = manifest.as_ref().and_then(|m| m.system.clone()).filter(|s| !s.is_empty()) {
            messages.insert(0, ChatMessage::new("system", sys));
        }
    }
    // Modelfile MESSAGE turns: inject once, right after the system prompt,
    // only on a fresh conversation (client sent just their first user turn).
    if let Some(mf) = manifest.as_ref().filter(|m| !m.messages.is_empty()) {
        if !has_prior_assistant(&messages) {
            let at = messages.iter().position(|m| m.role != "system").unwrap_or(messages.len());
            for (i, msg) in mf.messages.iter().enumerate() {
                messages.insert(at + i, ChatMessage::new(&msg.role, &msg.content));
            }
        }
    }
    let mut params = GenParams::chat(messages);
    params.sampling = ollama_options(&req.options);
    params.response_format = ollama_format(&req.format);
    params.think = req.think;
    respond(st, req.model, params, req.stream.unwrap_or(true), Kind::Chat).await
}

/// True when the client history already contains an assistant turn
/// (so stored few-shot turns would duplicate).
fn has_prior_assistant(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|m| m.role == "assistant")
}

fn to_store_messages(msgs: &[crate::modelfile::Message]) -> Vec<crate::store::Message> {
    msgs.iter()
        .map(|x| crate::store::Message { role: x.role.clone(), content: x.content.clone() })
        .collect()
}

// ---------- embeddings (real) ----------

fn inputs_of(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string())).collect(),
        Value::Null => vec![],
        other => vec![other.to_string()],
    }
}

#[derive(Deserialize)]
struct EmbedReq {
    model: String,
    #[serde(default)]
    input: Value,
}
async fn api_embed(State(st): State<AppState>, Json(req): Json<EmbedReq>) -> Response {
    if let Err(e) = ensure_model(&st, &req.model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    let t0 = Instant::now();
    match st.engine.embed(&req.model, inputs_of(&req.input)).await {
        Ok((vecs, ptok, load_ms)) => Json(json!({
            "model": req.model,
            "embeddings": vecs,
            "total_duration": ns(t0.elapsed().as_secs_f64() * 1e3),
            "load_duration": ns(load_ms),
            "prompt_eval_count": ptok,
        }))
        .into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

#[derive(Deserialize)]
struct EmbedLegacyReq {
    model: String,
    #[serde(default)]
    prompt: String,
}
async fn api_embeddings_legacy(State(st): State<AppState>, Json(req): Json<EmbedLegacyReq>) -> Response {
    if let Err(e) = ensure_model(&st, &req.model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    match st.engine.embed(&req.model, vec![req.prompt]).await {
        Ok((mut vecs, _, _)) => Json(json!({ "embedding": vecs.pop().unwrap_or_default() })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

// ---------- pull / create / copy / delete ----------

#[derive(Deserialize)]
struct PullReq {
    #[serde(alias = "name")]
    model: String,
    #[serde(default)]
    skip_verify: Option<bool>,
}
async fn api_pull(State(st): State<AppState>, Json(req): Json<PullReq>) -> impl IntoResponse {
    // Stream NDJSON progress like Ollama.
    let model = req.model.clone();
    let skip_verify = req.skip_verify.unwrap_or(false)
        || std::env::var("LLMON_SKIP_VERIFY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let store = st.store.clone();
    let s = async_stream::stream! {
        yield ndjson_line(&json!({"status": format!("pulling {model}")}));
        match do_pull(&store, &model, skip_verify).await {
            Ok(_) => yield ndjson_line(&json!({"status": "success"})),
            Err(e) => yield ndjson_line(&json!({"error": format!("{e:#}")})),
        }
    };
    ([("content-type", "application/x-ndjson")], axum::body::Body::from_stream(s)).into_response()
}

#[derive(Deserialize)]
struct PushReq {
    #[serde(alias = "name")]
    model: String,
    /// Destination `owner/repo[/file.gguf]` (Hugging Face).
    #[serde(default)]
    destination: Option<String>,
    #[serde(default)]
    insecure: Option<bool>,
}

/// `POST /api/push`: upload the model's GGUF to Hugging Face.
/// Streams NDJSON progress like Ollama. Needs `HF_TOKEN`/`HF_HUB_TOKEN`
/// with write access (Ollama instead needs an ollama.com signin).
async fn api_push(State(st): State<AppState>, Json(req): Json<PushReq>) -> impl IntoResponse {
    let model = req.model.clone();
    let destination = req.destination.clone();
    if req.insecure.unwrap_or(false) {
        tracing::warn!("--insecure accepted for OCI-registry compat; Hugging Face requires HTTPS");
    }
    let store = st.store.clone();
    let s = async_stream::stream! {
        yield ndjson_line(&json!({"status": format!("pushing {model}")}));
        match do_push(&store, &model, destination.as_deref()).await {
            Ok(msg) => yield ndjson_line(&json!({"status": msg})),
            Err(e) => yield ndjson_line(&json!({"error": format!("{e:#}")})),
        }
    };
    ([("content-type", "application/x-ndjson")], axum::body::Body::from_stream(s)).into_response()
}

pub async fn do_push(store: &Store, model: &str, destination: Option<&str>) -> anyhow::Result<String> {
    let m = store.resolve(model)?;
    let gguf = store.blob_path(&m.blob_digest);
    let (repo, mut filename) = match destination {
        Some(d) => split_push_dest(d)?,
        None => anyhow::bail!("destination owner/repo[/file.gguf] required"),
    };
    if filename.is_empty() {
        filename = format!("{}-{}.gguf", safe(&m.name), safe(&m.tag));
    }
    crate::registry::push_to_hf(&repo, &gguf, &filename, None::<fn(u64, Option<u64>)>).await
}

fn split_push_dest(dest: &str) -> anyhow::Result<(String, String)> {
    let dest = dest.strip_prefix("hf:").unwrap_or(dest);
    let parts: Vec<&str> = dest.split('/').collect();
    if parts.len() < 2 {
        anyhow::bail!("destination must be owner/repo[/file.gguf], got '{dest}'");
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let filename = if parts.len() > 2 { parts[2..].join("/") } else { String::new() };
    Ok((repo, filename))
}

pub async fn do_pull(store: &Store, model: &str, skip_verify: bool) -> anyhow::Result<String> {
    let verify = !skip_verify;
    let (name, tag) = Store::split_name(model);
    let (url, fname) = crate::registry::resolve_pull_url(&name).await?;
    let dest = store.blobs_dir().join(format!("{}-{}", safe(&name), safe(&fname)));
    std::fs::create_dir_all(store.blobs_dir())?;
    let (size, digest) = crate::registry::download_to(&url, &dest, verify, None::<fn(u64, Option<u64>)>).await?;
    // Point tag at blob (move into content-addressed name, keep file).
    let final_path = store.blob_path(&digest);
    if dest != final_path {
        if final_path.exists() {
            std::fs::remove_file(&dest)?;
        } else {
            std::fs::rename(&dest, &final_path)?;
        }
    }
    store.put_alias(
        &name,
        &tag,
        &crate::store::LocalManifest {
            name: name.clone(),
            tag: tag.clone(),
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
            created_at: crate::store::now_rfc3339(),
        },
    )?;
    tracing::debug!("verified {digest} in a single pass ({size} bytes, no re-read)");
    Ok(format!("success: pulled {name}:{tag}"))
}

fn safe(s: &str) -> String {
    s.replace(['/', '\\', ':', ' '], "-")
}

#[derive(Deserialize)]
struct CreateReq {
    model: String,
    #[serde(default)]
    modelfile: Option<String>,
    #[serde(default)]
    path: Option<String>,
}
async fn api_create(State(st): State<AppState>, Json(req): Json<CreateReq>) -> impl IntoResponse {
    let mf_text = if let Some(p) = req.path {
        match std::fs::read_to_string(&p) {
            Ok(t) => t,
            Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
        }
    } else if let Some(m) = req.modelfile {
        m
    } else {
        return err(StatusCode::BAD_REQUEST, "modelfile or path required");
    };
    match crate::modelfile::parse_str(&mf_text) {
        Ok(mf) => {
            // FROM local file -> import; FROM remote -> record alias.
            let (name, tag) = Store::split_name(&req.model);
            let from_p = std::path::Path::new(&mf.from);
            if from_p.exists() {
                match st.store.import_file(&name, &tag, from_p) {
                    Ok(mut m) => {
                        m.system = mf.system;
                        m.template = mf.template;
                        m.license = mf.license;
                        m.messages = to_store_messages(&mf.messages);
                        m.draft = mf.draft;
                        m.draft_max = mf.draft_max;
                        m.keep_alive = mf.keep_alive;
                        m.adapter = mf.adapter;
                        m.mmproj = mf.mmproj;
m.parameters = mf.parameters.clone();
                        let _ = st.store.put_alias(&name, &tag, &m);
                        Json(json!({"status": format!("created {name}:{tag}")})).into_response()
                    }
                    Err(e) => err(StatusCode::BAD_REQUEST, e.to_string()),
                }
            } else if let Ok(src) = st.store.resolve(&mf.from) {
                // FROM an already-local model: alias + metadata, no download.
                let mut m = src;
                m.name = name.clone();
                m.tag = tag.clone();
                m.system = mf.system.or(m.system);
                m.template = mf.template.or(m.template);
                if mf.draft.is_some() {
                    m.draft = mf.draft;
                    m.draft_max = mf.draft_max;
                if mf.keep_alive.is_some() {
                    m.keep_alive = mf.keep_alive;
                }
                }
                if mf.adapter.is_some() {
                    m.adapter = mf.adapter;
                }
                if mf.mmproj.is_some() {
                    m.mmproj = mf.mmproj;
                }
                // Inherit source parameters, Modelfile wins on conflict.
                m.parameters.extend(mf.parameters.clone());
                if !mf.messages.is_empty() {
                    m.messages = to_store_messages(&mf.messages);
                }
                let _ = st.store.put_alias(&name, &tag, &m);
                Json(json!({"status": format!("created {name}:{tag}")})).into_response()
            } else {
                // Remote FROM: pull then attach metadata.
                match do_pull(&st.store, &mf.from, false).await {
                    Ok(_) => {
                        if let Ok(src) = st.store.get(&mf.from) {
                            let mut m = src;
                            m.name = name.clone();
                            m.tag = tag.clone();
                            m.system = mf.system.or(m.system);
                            m.template = mf.template.or(m.template);
                            if mf.draft.is_some() {
                                m.draft = mf.draft;
                                m.draft_max = mf.draft_max;
                            if mf.keep_alive.is_some() {
                                m.keep_alive = mf.keep_alive;
                            }
                            }
                            if mf.adapter.is_some() {
                                m.adapter = mf.adapter;
                            }
                            if mf.mmproj.is_some() {
                                m.mmproj = mf.mmproj;
                            }
                            m.parameters.extend(mf.parameters.clone());
                            if !mf.messages.is_empty() {
                                m.messages = to_store_messages(&mf.messages);
                            }
                            let _ = st.store.put_alias(&name, &tag, &m);
                        }
                        Json(json!({"status": format!("created {name}:{tag}")})).into_response()
                    }
                    Err(e) => err(StatusCode::BAD_REQUEST, format!("FROM pull failed: {e:#}")),
                }
            }
        }
        Err(e) => err(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

#[derive(Deserialize)]
struct CopyReq {
    source: String,
    destination: String,
}
async fn api_copy(State(st): State<AppState>, Json(req): Json<CopyReq>) -> impl IntoResponse {
    match st.store.copy(&req.source, &req.destination) {
        Ok(_) => Json(json!({"status": "ok"})).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e.to_string()),
    }
}

async fn api_delete(State(st): State<AppState>, body: axum::body::Bytes) -> impl IntoResponse {
    // Accept model from JSON body (Ollama style: `model` or legacy `name`).
    let model: Option<String> = serde_json::from_slice::<Value>(&body).ok().and_then(|v| {
        v.get("model").or_else(|| v.get("name")).and_then(|m| m.as_str()).map(|s| s.to_string())
    });
    let Some(model) = model else {
        return err(StatusCode::BAD_REQUEST, "model required");
    };
    st.engine.unload(&model).await;
    match st.store.remove(&model) {
        Ok(true) => Json(json!({"status": "ok"})).into_response(),
        Ok(false) => err(StatusCode::NOT_FOUND, format!("{model} not found")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ---------- OpenAI-compatible (raw passthrough) ----------

async fn v1_models(State(st): State<AppState>) -> impl IntoResponse {
    let models = st.store.list().unwrap_or_default();
    Json(json!({
        "object": "list",
        "data": models.iter().map(|m| json!({
            "id": format!("{}:{}", m.name, m.tag),
            "object": "model",
            "created": 0,
            "owned_by": crate::branding::APP_NAME,
        })).collect::<Vec<_>>(),
    }))
}

/// Forward the OpenAI request body untouched to the backend and stream the
/// response bytes straight back (SSE or JSON).
async fn passthrough(st: AppState, mut body: Value, path: &str, embedding: bool) -> Response {
    let Some(model) = body.get("model").and_then(Value::as_str).map(String::from) else {
        return err(StatusCode::BAD_REQUEST, "model required");
    };
    if let Err(e) = ensure_model(&st, &model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    if !st.engine.is_real() {
        return stub_openai(st, body, path).await;
    }
    // Reuse the KV prefix across turns (multi-turn TTFT stays flat).
    if body.get("cache_prompt").is_none() && !embedding {
        body["cache_prompt"] = json!(true);
    }
    match st.engine.proxy(&model, path, &body, embedding).await {
        Ok((resp, guard)) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let ctype = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();
            let s = resp.bytes_stream().map(move |c| {
                let _keep = &guard; // runner stays "busy" until the body ends
                c.map_err(std::io::Error::other)
            });
            Response::builder()
                .status(status)
                .header("content-type", ctype)
                .header("cache-control", "no-cache")
                .body(axum::body::Body::from_stream(s))
                .unwrap_or_else(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "response build failed"))
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn v1_chat_completions(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(st, body, "/v1/chat/completions", false).await
}
async fn v1_completions(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(st, body, "/v1/completions", false).await
}
async fn v1_embeddings(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(st, body, "/v1/embeddings", true).await
}

/// `GET /v1/models/:model` — Ollama and vLLM both serve this per-model view.
async fn v1_model_show(State(st): State<AppState>, axum::extract::Path(model): axum::extract::Path<String>) -> Response {
    match st.store.resolve(&model) {
        Ok(m) => Json(json!({
            "id": format!("{}:{}", m.name, m.tag),
            "object": "model",
            "created": 0,
            "owned_by": crate::branding::APP_NAME,
            "size": m.size,
            "details": { "format": "gguf", "parameter_size": m.parameter_size, "quantization_level": m.quantization },
        }))
        .into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e.to_string()),
    }
}

/// Tokenizer endpoints. llama.cpp serves both; we forward `model` from either
/// query or body so callers can pick which runner to use.
async fn v1_tokenize(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    tokenizer_proxy(st, body, "/tokenize").await
}
async fn v1_detokenize(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    tokenizer_proxy(st, body, "/detokenize").await
}

async fn tokenizer_proxy(st: AppState, body: Value, path: &str) -> Response {
    let Some(model) = body.get("model").and_then(Value::as_str).map(String::from) else {
        return err(StatusCode::BAD_REQUEST, "model required");
    };
    if let Err(e) = ensure_model(&st, &model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    if !st.engine.is_real() {
        // Stub mode: naive whitespace tokenization so clients still work.
        if path.ends_with("tokenize") {
            let content = body.get("content").and_then(Value::as_str).unwrap_or("");
            let tokens: Vec<usize> = content.bytes().map(|b| b as usize).collect();
            return Json(json!({ "tokens": tokens })).into_response();
        }
        let tokens = body.get("tokens").and_then(Value::as_array).cloned().unwrap_or_default();
        let content: String = tokens.iter().filter_map(Value::as_u64).map(|t| (t as u8) as char).collect();
        return Json(json!({ "content": content })).into_response();
    }
    proxy_path(st, &model, path, &body, false).await
}

/// Rerank/score endpoints: llama.cpp needs `--reranking`; we pass through and
/// forward the backend's own error when unsupported.
async fn v1_rerank(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    if let Err(e) = rerank_ready(&st) {
        return err(StatusCode::BAD_REQUEST, e);
    }
    passthrough_path(st, body, "/v1/rerank").await
}

/// Anthropic `/v1/messages` passthrough (llama.cpp implements the translation).
async fn v1_messages(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough_path(st, body, "/v1/messages").await
}

async fn v1_messages_count_tokens(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough_path(st, body, "/v1/messages/count_tokens").await
}

/// Dynamic LoRA: `POST /v1/load_lora_adapter {model, adapter[, scale]}`.
async fn v1_load_lora(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    let Some(model) = body.get("model").and_then(Value::as_str).map(String::from) else {
        return err(StatusCode::BAD_REQUEST, "model required");
    };
    let Some(adapter) = body.get("adapter").and_then(Value::as_str).map(String::from) else {
        return err(StatusCode::BAD_REQUEST, "adapter required (store alias or file path)");
    };
    if let Err(e) = ensure_model(&st, &model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    // Validate the adapter resolves before touching anything.
    let ok = st.store.gguf_path(&adapter).is_ok() || std::path::Path::new(&adapter).is_file();
    if !ok {
        return err(StatusCode::NOT_FOUND, format!("adapter '{adapter}' not found in store or on disk"));
    }
    match st.store.resolve(&model) {
        Ok(mut m) => {
            m.adapter = Some(adapter.clone());
            let key = (m.name.clone(), m.tag.clone());
            if st.store.put_alias(&key.0, &key.1, &m).is_err() {
                return err(StatusCode::INTERNAL_SERVER_ERROR, "could not update manifest");
            }
            st.engine.unload(&model).await; // next request respawns with --lora
            Json(json!({ "status": "ok", "model": model, "adapter": adapter })).into_response()
        }
        Err(e) => err(StatusCode::NOT_FOUND, e.to_string()),
    }
}

/// `POST /v1/unload_lora_adapter {model}`: clear the adapter, respawn clean.
async fn v1_unload_lora(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    let Some(model) = body.get("model").and_then(Value::as_str).map(String::from) else {
        return err(StatusCode::BAD_REQUEST, "model required");
    };
    match st.store.resolve(&model) {
        Ok(mut m) => {
            m.adapter = None;
            let key = (m.name.clone(), m.tag.clone());
            if st.store.put_alias(&key.0, &key.1, &m).is_err() {
                return err(StatusCode::INTERNAL_SERVER_ERROR, "could not update manifest");
            }
            st.engine.unload(&model).await;
            Json(json!({ "status": "ok", "model": model })).into_response()
        }
        Err(e) => err(StatusCode::NOT_FOUND, e.to_string()),
    }
}

/// `GET /tokenizer_info?model=...`: bos/eos ids and vocab size from the
/// backend `/props` (vLLM-compatible shape).
async fn v1_tokenizer_info(State(st): State<AppState>, axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    let Some(model) = q.get("model").cloned().or_else(|| first_model(&st)) else {
        return err(StatusCode::BAD_REQUEST, "model required (no local models)");
    };
    if let Err(e) = ensure_model(&st, &model) {
        return err(StatusCode::NOT_FOUND, e);
    }
    if !st.engine.is_real() {
        return err(StatusCode::NOT_IMPLEMENTED, "tokenizer info requires the llama.cpp backend");
    }
    // Include numeric ids when the chat template exposes them is overkill;
    // report what /props actually carries: token strings + model facts.
    match st.engine.proxy_get(&model, "/props", false).await {
        Ok((resp, _guard)) => {
            let v: Value = resp.json().await.unwrap_or_default();
            Json(json!({
                "model": model,
                "bos_token": v.get("bos_token"),
                "eos_token": v.get("eos_token"),
                "model_alias": v.get("model_alias"),
                "model_ftype": v.get("model_ftype"),
                "modalities": v.get("modalities"),
                "has_chat_template": v.get("chat_template").and_then(|t| t.as_str()).map(|t| !t.is_empty()).unwrap_or(false),
            }))
            .into_response()
        }
        Err(e) => err(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    }
}

fn first_model(st: &AppState) -> Option<String> {
    st.store.list().ok()?.into_iter().next().map(|m| format!("{}:{}", m.name, m.tag))
}

/// Audio transcription/translation: multipart bodies can't ride the JSON
/// proxy, so forward bytes + content-type verbatim. The backend itself
/// decides support (audio-capable multimodal model required); its error is
/// forwarded untouched instead of faked.
async fn v1_audio_transcriptions(
    State(st): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    v1_audio_proxy(st, q, headers, body, "/v1/audio/transcriptions").await
}

async fn v1_audio_translations(
    State(st): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    v1_audio_proxy(st, q, headers, body, "/v1/audio/translations").await
}

async fn v1_audio_proxy(
    st: AppState,
    q: std::collections::HashMap<String, String>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
    path: &str,
) -> Response {
    let model = match ensure_or_default(&st, q.get("model").map(String::as_str).unwrap_or("default")) {
        Ok(m) => m,
        Err(e) => return err(StatusCode::NOT_FOUND, e),
    };
    if !st.engine.is_real() {
        return err(StatusCode::NOT_IMPLEMENTED, "audio transcription requires the llama.cpp backend");
    }
    let ctype = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();
    match st.engine.proxy_bytes(&model, path, ctype, body).await {
        Ok((resp, guard)) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let s = resp.bytes_stream().map(move |c| {
                let _keep = &guard;
                c.map_err(std::io::Error::other)
            });
            Response::builder()
                .status(status)
                .header("content-type", "application/json")
                .body(axum::body::Body::from_stream(s))
                .unwrap_or_else(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "response build failed"))
        }
        Err(e) => err(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    }
}

async fn v1_responses_input_tokens(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough_path(st, body, "/v1/responses/input_tokens").await
}

/// OpenAI `/v1/responses` passthrough.
async fn v1_responses(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough_path(st, body, "/v1/responses").await
}

/// Stub-mode OpenAI responses (no backend installed).
async fn stub_openai(st: AppState, body: Value, path: &str) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or_default().to_string();
    if path.ends_with("embeddings") {
        let inputs = inputs_of(body.get("input").unwrap_or(&Value::Null));
        let (vecs, _, _) = st.engine.embed(&model, inputs).await.unwrap_or_default();
        let data: Vec<Value> = vecs
            .into_iter()
            .enumerate()
            .map(|(i, e)| json!({ "object": "embedding", "index": i, "embedding": e }))
            .collect();
        return Json(json!({ "object": "list", "data": data, "model": model })).into_response();
    }
    let prompt = body
        .get("messages")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|m| m.get("content").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"))
        .or_else(|| body.get("prompt").and_then(Value::as_str).map(String::from))
        .unwrap_or_default();
    let mut text = String::new();
    let mut ev = st.engine.generate(model.clone(), GenParams::raw(prompt));
    while let Some(e) = ev.next().await {
        if let Event::Token(t) = e {
            text.push_str(&t);
        }
    }
    Json(json!({
        "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "text": text, "finish_reason": "stop" }],
    }))
    .into_response()
}

fn ensure_model(st: &AppState, model: &str) -> Result<(), String> {
    // Real backend: fast memory check first, fallback to store resolve
    if st.engine.is_real() {
        if st.engine.has_loaded(model) {
            return Ok(());
        }
        return st.store.resolve(model).map(|_| ()).map_err(|e| e.to_string());
    }
    // Stub backend: allow ad-hoc names (tests/bench/demos without weights).
    Ok(())
}

/// Anthropic/Responses style: callers may omit `model` or pass `"default"`.
/// Resolve to a store model when possible, else to the first local model.
fn ensure_or_default(st: &AppState, model: &str) -> Result<String, String> {
    if model != "default" && model != "claude" {
        ensure_model(st, model)?;
        return Ok(model.to_string());
    }
    let list = st.store.list().map_err(|e| e.to_string())?;
    let m = list
        .first()
        .ok_or_else(|| "no local models — run `pull <model>` first".to_string())?;
    Ok(format!("{}:{}", m.name, m.tag))
}

fn rerank_ready(st: &AppState) -> Result<(), String> {
    if !st.engine.is_real() {
        return Err("reranking requires the llama.cpp backend (not available in stub mode)".into());
    }
    Ok(())
}

/// Proxy a JSON body to an arbitrary backend path, streaming the response back.
/// Unlike `passthrough` this accepts paths outside the OpenAI surface
/// (`/tokenize`, `/v1/messages`, ...) and does not inject `cache_prompt`.
async fn passthrough_path(st: AppState, mut body: Value, path: &str) -> Response {
    // Anthropic/Responses callers may omit `model` or pass `"default"`;
    // resolve to a concrete local model before forwarding.
    let raw_model = body.get("model").and_then(Value::as_str).map(String::from);
    let resolved = match raw_model.as_deref() {
        Some(m) if m != "default" && m != "claude" => {
            if let Err(e) = ensure_model(&st, m) {
                return err(StatusCode::NOT_FOUND, e);
            }
            m.to_string()
        }
        _ => match ensure_or_default(&st, "default") {
            Ok(m) => m,
            Err(e) => return err(StatusCode::NOT_FOUND, e),
        },
    };
    body["model"] = json!(resolved);
    if !st.engine.is_real() {
        return err(StatusCode::NOT_IMPLEMENTED, "endpoint requires the llama.cpp backend");
    }
    proxy_path(st, &resolved, path, &body, false).await
}

/// Shared low-level proxy for arbitrary paths (no `cache_prompt` injection).
async fn proxy_path(st: AppState, model: &str, path: &str, body: &Value, embedding: bool) -> Response {
    match st.engine.proxy(model, path, body, embedding).await {
        Ok((resp, guard)) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let ctype = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();
            let s = resp.bytes_stream().map(move |c| {
                let _keep = &guard; // runner stays busy until the body ends
                c.map_err(std::io::Error::other)
            });
            Response::builder()
                .status(status)
                .header("content-type", ctype)
                .header("cache-control", "no-cache")
                .body(axum::body::Body::from_stream(s))
                .unwrap_or_else(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "response build failed"))
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_keep_alive_forms() {
        assert_eq!(parse_keep_alive(&Some(json!(300))), Some(300));
        assert_eq!(parse_keep_alive(&Some(json!("5m"))), Some(300));
        assert_eq!(parse_keep_alive(&Some(json!("1h"))), Some(3600));
        assert_eq!(parse_keep_alive(&Some(json!("90s"))), Some(90));
        assert_eq!(parse_keep_alive(&Some(json!("0"))), Some(0));
        assert_eq!(parse_keep_alive(&Some(json!("-1"))), Some(-1));
        assert_eq!(parse_keep_alive(&None), None);
    }

    #[test]
    fn duration_edge_cases() {
        assert_eq!(parse_duration("2h"), Some(7200));
        assert_eq!(parse_duration("45"), Some(45));
        assert_eq!(parse_duration("bad"), None);
    }
}
