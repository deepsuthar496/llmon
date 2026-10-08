//! `run`: one-shot or interactive chat.
//!
//! Mirrors Ollama's interactive CLI conventions (`>>> ` prompt, `/bye`, history).
//! - Primary prompt `>>> ` and multiline prompt `... `
//! - Multiline blocks delimited by `"""`
//! - Slash commands:
//!     /set parameter <name> <value>  (temperature, top_p, top_k, seed, num_predict, etc.)
//!     /set system <string>           (sets session system prompt)
//!     /set verbose / /set quiet      (show/hide generation stats)
//!     /set format json / /set noformat
//!     /set think / /set nothink
//!     /set history / /set nohistory
//!     /set wordwrap / /set nowordwrap
//!     /show info                     (architecture, parameters, size, digest)
//!     /show system                   (current system prompt)
//!     /show parameters               (current sampling options)
//!     /show modelfile                (Modelfile representation)
//!     /show license                  (model license)
//!     /load <model>                  (switch active model)
//!     /save <model>                  (save session as new model tag)
//!     /clear                         (reset conversation history)
//!     /bye, /exit, /quit             (exit interactive shell)
//!     /?, /help, /? shortcuts        (help text & keyboard shortcuts)
//! - History persistence to ~/.llmon/history
//! - Real-time streaming output
//! - Clean Ctrl+C handling to interrupt generation without quitting

use std::io::Write;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use rustyline::error::ReadlineError;
use serde_json::{json, Map, Value};

use crate::engine::{ChatMessage, Engine, Event, EventStream, GenParams, Stats};
use crate::store::Store;

#[derive(Clone, Debug)]
pub struct SessionOptions {
    pub system: Option<String>,
    pub options: Map<String, Value>,
    pub format: Option<String>,
    pub think: Option<bool>,
    pub verbose: bool,
    pub wordwrap: bool,
    pub history: bool,
    /// Ollama `keep_alive` for daemon requests (e.g. `"5m"`, `"0"`).
    pub keep_alive: Option<String>,
    /// Hide thinking output (Ollama `--hidethinking`).
    pub hide_thinking: bool,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            system: None,
            options: Map::new(),
            format: None,
            think: None,
            verbose: false,
            wordwrap: true,
            history: true,
            keep_alive: None,
            hide_thinking: false,
        }
    }
}

pub enum Backend {
    Daemon { base: String, client: reqwest::Client },
    Local(Engine),
}

impl Backend {
    /// Prefer a running daemon (fast: model already resident).
    pub async fn connect(base_url: &str, engine: Engine) -> Self {
        let client = reqwest::Client::builder().no_proxy().tcp_nodelay(true).build().unwrap_or_default();
        let up = client
            .get(format!("{base_url}/health"))
            .timeout(Duration::from_millis(300))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        if up {
            Backend::Daemon { base: base_url.to_string(), client }
        } else {
            Backend::Local(engine)
        }
    }

    #[allow(dead_code)]
    pub fn describe(&self) -> String {
        match self {
            Backend::Daemon { base, .. } => format!("daemon {base}"),
            Backend::Local(e) => format!("in-process {}", e.backend_name()),
        }
    }

    pub async fn chat(&self, model: &str, messages: &[ChatMessage], opts: &SessionOptions) -> EventStream {
        match self {
            Backend::Local(engine) => {
                let mut params = GenParams::chat(messages.to_vec());
                params.sampling = opts.options.clone();
                if let Some(f) = &opts.format {
                    if f == "json" {
                        params.response_format = Some(json!({ "type": "json_object" }));
                    }
                }
                params.think = opts.think;
                engine.generate(model.to_string(), params)
            }
            Backend::Daemon { base, client } => {
                let mut body = json!({
                    "model": model,
                    "messages": messages,
                    "stream": true,
                });
                if !opts.options.is_empty() {
                    body["options"] = json!(opts.options);
                }
                if let Some(f) = &opts.format {
                    body["format"] = json!(f);
                }
                if let Some(t) = opts.think {
                    body["think"] = json!(t);
                }
                if let Some(ka) = &opts.keep_alive {
                    // Numeric strings become numbers, else keep as duration string.
                    if let Ok(n) = ka.parse::<u64>() {
                        body["keep_alive"] = json!(n);
                    } else {
                        body["keep_alive"] = json!(ka);
                    }
                }
                let req = client.post(format!("{base}/api/chat")).json(&body);
                Box::pin(async_stream::stream! {
                    let resp = match req.send().await {
                        Ok(r) => r,
                        Err(e) => { yield Event::Error(e.to_string()); return; }
                    };
                    if !resp.status().is_success() {
                        let code = resp.status();
                        let t = resp.text().await.unwrap_or_default();
                        yield Event::Error(format!("{code}: {t}"));
                        return;
                    }
                    let mut body = resp.bytes_stream();
                    let mut buf: Vec<u8> = vec![];
                    while let Some(chunk) = body.next().await {
                        let Ok(chunk) = chunk else { break };
                        buf.extend_from_slice(&chunk);
                        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                            let line: Vec<u8> = buf.drain(..=pos).collect();
                            let Ok(v) = serde_json::from_slice::<Value>(&line) else { continue };
                            if let Some(e) = v.get("error") {
                                yield Event::Error(e.as_str().unwrap_or_default().to_string());
                                return;
                            }
                            if let Some(t) = v.pointer("/message/thinking").and_then(Value::as_str) {
                                if !t.is_empty() { yield Event::Thinking(t.to_string()); }
                            }
                            if let Some(t) = v.pointer("/message/content").and_then(Value::as_str) {
                                if !t.is_empty() { yield Event::Token(t.to_string()); }
                            }
                            if v.get("done").and_then(Value::as_bool) == Some(true) {
                                let n = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                                yield Event::Done(Stats {
                                    prompt_tokens: n("prompt_eval_count") as u64,
                                    prompt_ms: n("prompt_eval_duration") / 1e6,
                                    eval_tokens: n("eval_count") as u64,
                                    eval_ms: n("eval_duration") / 1e6,
                                    load_ms: n("load_duration") / 1e6,
                                    total_ms: n("total_duration") / 1e6,
                                    done_reason: v.get("done_reason").and_then(Value::as_str).unwrap_or("stop").into(),
                                });
                                return;
                            }
                        }
                    }
                })
            }
        }
    }
}

/// Stream one assistant turn to stdout; returns the full reply.
async fn print_turn(
    backend: &Backend,
    model: &str,
    history: &[ChatMessage],
    opts: &SessionOptions,
) -> Result<String> {
    let mut s = backend.chat(model, history, opts).await;
    let mut out = std::io::stdout().lock();
    let mut reply = String::new();
    let mut thinking = false;
    let mut interrupted = false;

    loop {
        tokio::select! {
            ev = s.next() => {
                let Some(ev) = ev else { break; };
                match ev {
                    Event::Thinking(t) => {
                        if !opts.hide_thinking {
                            if !thinking {
                                write!(out, "\x1b[2mThinking...\n")?;
                                thinking = true;
                            }
                            write!(out, "{t}")?;
                            out.flush()?;
                        }
                    }
                    Event::Token(t) => {
                        if thinking && !opts.hide_thinking {
                            write!(out, "\n...done thinking.\x1b[0m\n\n")?;
                            thinking = false;
                        }
                        write!(out, "{t}")?;
                        out.flush()?;
                        reply.push_str(&t);
                    }
                    Event::Done(st) => {
                        writeln!(out)?;
                        if opts.verbose {
                            print_stats(&st);
                        }
                    }
                    Event::Error(e) => {
                        writeln!(out)?;
                        anyhow::bail!("{e}");
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                writeln!(out, "\n\x1b[2m(interrupted)\x1b[0m")?;
                interrupted = true;
                break;
            }
        }
    }

    if thinking {
        write!(out, "\x1b[0m")?;
        out.flush()?;
    }
    if interrupted && reply.is_empty() {
        anyhow::bail!("interrupted");
    }
    Ok(reply)
}

fn print_stats(s: &Stats) {
    let rate = |n: u64, ms: f64| if ms > 0.0 { n as f64 / (ms / 1e3) } else { 0.0 };
    eprintln!("total duration:       {:.2?}", Duration::from_secs_f64(s.total_ms / 1e3));
    eprintln!("load duration:        {:.2?}", Duration::from_secs_f64(s.load_ms / 1e3));
    eprintln!("prompt eval count:    {} token(s)", s.prompt_tokens);
    eprintln!("prompt eval duration: {:.2?}", Duration::from_secs_f64(s.prompt_ms / 1e3));
    eprintln!("prompt eval rate:     {:.2} tokens/s", rate(s.prompt_tokens, s.prompt_ms));
    eprintln!("eval count:           {} token(s)", s.eval_tokens);
    eprintln!("eval duration:        {:.2?}", Duration::from_secs_f64(s.eval_ms / 1e3));
    eprintln!("eval rate:            {:.2} tokens/s", rate(s.eval_tokens, s.eval_ms));
}

fn print_usage() {
    eprintln!("Available Commands:");
    eprintln!("  /set            Set session variables");
    eprintln!("  /show           Show model information");
    eprintln!("  /load <model>   Load a session or model");
    eprintln!("  /save <model>   Save your current session");
    eprintln!("  /clear          Clear session context");
    eprintln!("  /bye            Exit");
    eprintln!("  /?, /help       Help for a command");
    eprintln!("  /? shortcuts    Help for keyboard shortcuts");
    eprintln!();
    eprintln!("Use \"\"\" to begin a multi-line message.");
    eprintln!();
}

fn print_usage_set() {
    eprintln!("Available Commands:");
    eprintln!("  /set parameter ...     Set a parameter");
    eprintln!("  /set system <string>   Set system message");
    eprintln!("  /set history           Enable history");
    eprintln!("  /set nohistory         Disable history");
    eprintln!("  /set wordwrap          Enable wordwrap");
    eprintln!("  /set nowordwrap        Disable wordwrap");
    eprintln!("  /set format json       Enable JSON mode");
    eprintln!("  /set noformat          Disable formatting");
    eprintln!("  /set verbose           Show LLM stats");
    eprintln!("  /set quiet             Disable LLM stats");
    eprintln!("  /set think             Enable thinking");
    eprintln!("  /set nothink           Disable thinking");
    eprintln!();
}

fn print_usage_parameters() {
    eprintln!("Available Parameters:");
    eprintln!("  /set parameter seed <int>             Random number seed");
    eprintln!("  /set parameter num_predict <int>      Max number of tokens to predict");
    eprintln!("  /set parameter top_k <int>            Pick from top k num of tokens");
    eprintln!("  /set parameter top_p <float>          Pick token based on sum of probabilities");
    eprintln!("  /set parameter min_p <float>          Pick token based on top token probability * min_p");
    eprintln!("  /set parameter temperature <float>    Set creativity level");
    eprintln!("  /set parameter repeat_penalty <float> How strongly to penalize repetitions");
    eprintln!("  /set parameter repeat_last_n <int>    Set how far back to look for repetitions");
    eprintln!("  /set parameter stop <string> ...      Set the stop parameters");
    eprintln!();
}

fn print_usage_show() {
    eprintln!("Available Commands:");
    eprintln!("  /show info         Show details for this model");
    eprintln!("  /show license      Show model license");
    eprintln!("  /show modelfile    Show Modelfile for this model");
    eprintln!("  /show parameters   Show parameters for this model");
    eprintln!("  /show system       Show system message");
    eprintln!();
}

fn print_usage_shortcuts() {
    eprintln!("Available keyboard shortcuts:");
    eprintln!("  Ctrl + a            Move to the beginning of the line (Home)");
    eprintln!("  Ctrl + e            Move to the end of the line (End)");
    eprintln!("  Ctrl + k            Delete the sentence after the cursor");
    eprintln!("  Ctrl + u            Delete the sentence before the cursor");
    eprintln!("  Ctrl + w            Delete the word before the cursor");
    eprintln!("  Ctrl + l            Clear the screen");
    eprintln!("  Ctrl + c            Interrupt generation / cancel current input");
    eprintln!("  Ctrl + d            Exit (/bye)");
    eprintln!();
}

pub async fn one_shot(backend: &Backend, model: &str, prompt: &str, opts: &SessionOptions) -> Result<()> {
    print_turn(backend, model, &[ChatMessage::new("user", prompt)], opts).await?;
    Ok(())
}

#[derive(PartialEq)]
enum MultilineState {
    None,
    Prompt,
    System,
}

pub async fn chat_loop(
    backend: &Backend,
    store: Store,
    initial_model: &str,
    verbose: bool,
    opts: SessionOptions,
) -> Result<()> {
    let mut rl = rustyline::DefaultEditor::new()?;
    let history_path = dirs::home_dir().map(|h| h.join(".llmon").join("history"));
    if let Some(ref p) = history_path {
        let _ = rl.load_history(p);
    }

    let mut model = initial_model.to_string();
    let mut opts = SessionOptions { verbose, ..opts };
    let mut history: Vec<ChatMessage> = vec![];
    let mut multiline = MultilineState::None;
    let mut sb = String::new();


    loop {
        let prompt_str = if multiline != MultilineState::None { "... " } else { ">>> " };
        match rl.readline(prompt_str) {
            Ok(line) => {
                let trimmed = line.trim();

                // Multiline accumulation
                if multiline != MultilineState::None {
                    if let Some(rest) = line.strip_suffix("\"\"\"") {
                        sb.push_str(rest);
                        let final_content = sb.trim().to_string();
                        sb.clear();
                        let state = multiline;
                        multiline = MultilineState::None;

                        if state == MultilineState::System {
                            opts.system = Some(final_content.clone());
                            if let Some(first) = history.first_mut() {
                                if first.role == "system" {
                                    first.content = final_content;
                                } else {
                                    history.insert(0, ChatMessage::new("system", final_content));
                                }
                            } else {
                                history.push(ChatMessage::new("system", final_content));
                            }
                            println!("Set system message.");
                            continue;
                        }

                        // Send multiline prompt
                        if !final_content.is_empty() {
                            if opts.history {
                                let _ = rl.add_history_entry(&format!("\"\"\"{final_content}\"\"\""));
                                if let Some(ref p) = history_path {
                                    if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
                                    let _ = rl.save_history(p);
                                }
                            }
                            history.push(ChatMessage::new("user", final_content));
                            match print_turn(backend, &model, &history, &opts).await {
                                Ok(reply) => history.push(ChatMessage::new("assistant", reply)),
                                Err(e) => {
                                    history.pop();
                                    if e.to_string() != "interrupted" {
                                        eprintln!("error: {e:#}");
                                    }
                                }
                            }
                        }
                        continue;
                    } else {
                        sb.push_str(&line);
                        sb.push('\n');
                        continue;
                    }
                }

                // Check starting of multiline
                if let Some(after) = trimmed.strip_prefix("\"\"\"") {
                    if let Some(inside) = after.strip_suffix("\"\"\"") {
                        // Complete inline """message"""
                        let content = inside.trim().to_string();
                        if !content.is_empty() {
                            if opts.history {
                                let _ = rl.add_history_entry(trimmed);
                                if let Some(ref p) = history_path {
                                    if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
                                    let _ = rl.save_history(p);
                                }
                            }
                            history.push(ChatMessage::new("user", content));
                            match print_turn(backend, &model, &history, &opts).await {
                                Ok(reply) => history.push(ChatMessage::new("assistant", reply)),
                                Err(e) => {
                                    history.pop();
                                    if e.to_string() != "interrupted" {
                                        eprintln!("error: {e:#}");
                                    }
                                }
                            }
                        }
                        continue;
                    } else {
                        sb.push_str(after);
                        sb.push('\n');
                        multiline = MultilineState::Prompt;
                        continue;
                    }
                }

                if trimmed.is_empty() {
                    continue;
                }

                if opts.history {
                    let _ = rl.add_history_entry(trimmed);
                    if let Some(ref p) = history_path {
                        if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
                        let _ = rl.save_history(p);
                    }
                }

                // Slash command routing
                if trimmed.starts_with('/') {
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    match parts.first().copied().unwrap_or("") {
                        "/bye" | "/exit" | "/quit" => break,
                        "/?" | "/help" => {
                            if parts.get(1).copied() == Some("shortcuts") {
                                print_usage_shortcuts();
                            } else {
                                print_usage();
                            }
                        }
                        "/clear" => {
                            history.clear();
                            if let Some(sys) = &opts.system {
                                history.push(ChatMessage::new("system", sys));
                            }
                            println!("Cleared session context");
                        }
                        "/set" => {
                            match parts.get(1).copied() {
                                Some("verbose") => { opts.verbose = true; println!("Set 'verbose' mode."); }
                                Some("quiet") => { opts.verbose = false; println!("Set 'quiet' mode."); }
                                Some("wordwrap") => { opts.wordwrap = true; println!("Set 'wordwrap' mode."); }
                                Some("nowordwrap") => { opts.wordwrap = false; println!("Set 'nowordwrap' mode."); }
                                Some("history") => { opts.history = true; println!("History enabled."); }
                                Some("nohistory") => { opts.history = false; println!("History disabled."); }
                                Some("think") => { opts.think = Some(true); println!("Set 'think' mode."); }
                                Some("nothink") => { opts.think = Some(false); println!("Set 'nothink' mode."); }
                                Some("format") => {
                                    if parts.get(2).copied() == Some("json") {
                                        opts.format = Some("json".into());
                                        println!("Set format to 'json' mode.");
                                    } else {
                                        println!("Invalid or missing format. For 'json' mode use '/set format json'");
                                    }
                                }
                                Some("noformat") => { opts.format = None; println!("Disabled format."); }
                                Some("parameter") => {
                                    if parts.len() < 4 {
                                        print_usage_parameters();
                                    } else {
                                        let pname = parts[2];
                                        let pval = parts[3..].join(" ");
                                        let val: Value = if let Ok(i) = pval.parse::<i64>() {
                                            json!(i)
                                        } else if let Ok(f) = pval.parse::<f64>() {
                                            json!(f)
                                        } else {
                                            json!(pval)
                                        };
                                        opts.options.insert(pname.to_string(), val);
                                        println!("Set parameter '{pname}' to '{pval}'");
                                    }
                                }
                                Some("system") => {
                                    if parts.len() < 3 {
                                        print_usage_set();
                                    } else {
                                        let sys_arg = parts[2..].join(" ");
                                        if let Some(after) = sys_arg.strip_prefix("\"\"\"") {
                                            if let Some(inside) = after.strip_suffix("\"\"\"") {
                                                let content = inside.trim().to_string();
                                                opts.system = Some(content.clone());
                                                if let Some(first) = history.first_mut() {
                                                    if first.role == "system" { first.content = content; }
                                                    else { history.insert(0, ChatMessage::new("system", content)); }
                                                } else {
                                                    history.push(ChatMessage::new("system", content));
                                                }
                                                println!("Set system message.");
                                            } else {
                                                sb.push_str(after);
                                                sb.push('\n');
                                                multiline = MultilineState::System;
                                            }
                                        } else {
                                            opts.system = Some(sys_arg.clone());
                                            if let Some(first) = history.first_mut() {
                                                if first.role == "system" { first.content = sys_arg; }
                                                else { history.insert(0, ChatMessage::new("system", sys_arg)); }
                                            } else {
                                                history.push(ChatMessage::new("system", sys_arg));
                                            }
                                            println!("Set system message.");
                                        }
                                    }
                                }
                                _ => print_usage_set(),
                            }
                        }
                        "/show" => {
                            match parts.get(1).copied() {
                                Some("info") => {
                                    match store.resolve(&model) {
                                        Ok(m) => {
                                            println!("Model");
                                            println!("  name            {}", m.name);
                                            println!("  tag             {}", m.tag);
                                            println!("  size            {:.1} MB", m.size as f64 / 1e6);
                                            println!("  digest          {}", m.blob_digest);
                                            if let Some(q) = m.quantization { println!("  quantization    {q}"); }
                                            if let Some(p) = m.parameter_size { println!("  parameters      {p}"); }
                                            if let Some(v) = m.verified { println!("  verified        {v}"); }
                                        }
                                        Err(_) => println!("Model: {model} (running in resident engine)"),
                                    }
                                }
                                Some("license") => {
                                    match store.resolve(&model) {
                                        Ok(m) => {
                                            if let Some(lic) = m.license { println!("{lic}"); }
                                            else { println!("No license was specified for this model."); }
                                        }
                                        Err(_) => println!("No license was specified for this model."),
                                    }
                                }
                                Some("parameters") => {
                                    if opts.options.is_empty() {
                                        println!("No session parameters configured. Use '/set parameter <name> <value>'");
                                    } else {
                                        println!("User defined parameters:");
                                        for (k, v) in &opts.options {
                                            println!("  {k:<16} {v}");
                                        }
                                    }
                                }
                                Some("system") => {
                                    if let Some(s) = &opts.system {
                                        println!("System message:\n  {s}");
                                    } else {
                                        println!("No system message set. Use '/set system <message>'");
                                    }
                                }
                                Some("modelfile") => {
                                    println!("# Modelfile generated by llmon");
                                    println!("FROM {model}");
                                    if let Some(s) = &opts.system {
                                        println!("SYSTEM \"\"\"{s}\"\"\"");
                                    }
                                    for (k, v) in &opts.options {
                                        println!("PARAMETER {k} {v}");
                                    }
                                }
                                _ => print_usage_show(),
                            }
                        }
                        "/load" => {
                            if parts.len() != 2 {
                                println!("Usage:\n  /load <modelname>");
                            } else {
                                let target = parts[1];
                                match store.resolve(target) {
                                    Ok(_) => {
                                        model = target.to_string();
                                        history.clear();
                                        if let Some(sys) = &opts.system {
                                            history.push(ChatMessage::new("system", sys));
                                        }
                                        println!("Loading model '{model}'");
                                    }
                                    Err(_) => println!("Couldn't find model '{target}' (run `llmon pull {target}` first)"),
                                }
                            }
                        }
                        "/save" => {
                            if parts.len() != 2 {
                                println!("Usage:\n  /save <modelname>");
                            } else {
                                let dest = parts[1];
                                let (n, t) = Store::split_name(dest);
                                match store.resolve(&model) {
                                    Ok(mut m) => {
                                        m.name = n.clone();
                                        m.tag = t.clone();
                                        if let Some(s) = &opts.system { m.system = Some(s.clone()); }
                                        let _ = store.put_alias(&n, &t, &m);
                                        println!("Created new model '{dest}'");
                                    }
                                    Err(e) => println!("error saving model: {e:#}"),
                                }
                            }
                        }
                        other => {
                            println!("Unknown command '{other}'. Type /? for help");
                        }
                    }
                    continue;
                }

                // Regular chat message
                history.push(ChatMessage::new("user", trimmed));
                match print_turn(backend, &model, &history, &opts).await {
                    Ok(reply) => history.push(ChatMessage::new("assistant", reply)),
                    Err(e) => {
                        history.pop();
                        if e.to_string() != "interrupted" {
                            eprintln!("error: {e:#}");
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                if multiline != MultilineState::None {
                    multiline = MultilineState::None;
                    sb.clear();
                    println!("(input canceled)");
                } else {
                    println!("\nUse Ctrl + d or /bye to exit.");
                }
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(e) => {
                eprintln!("readline error: {e}");
                break;
            }
        }
    }

    if opts.history {
        if let Some(ref p) = history_path {
            if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
            let _ = rl.save_history(p);
        }
    }

    Ok(())
}
