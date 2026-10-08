//! Minimal Modelfile support (`FROM` + key directives).
//! Format (subset of Ollama Modelfile, enough for real use):
//! ```text
//! FROM ./model.gguf | hf:repo/file.gguf | registry:ns/name:tag
//! PARAMETER temperature 0.7
//! PARAMETER num_ctx 4096
//! SYSTEM """You are helpful."""
//! TEMPLATE """{{ .System }} {{ .Prompt }}"""
//! MESSAGE user Hello there
//! MESSAGE assistant Hi! How can I help?
//! LICENSE """..."""
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

/// A few-shot turn (`MESSAGE <role> <content>`), Ollama-compatible roles.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Default, Clone)]
pub struct Modelfile {
    pub from: String,
    pub parameters: HashMap<String, String>,
    pub system: Option<String>,
    pub template: Option<String>,
    pub license: Option<String>,
    /// `MESSAGE` turns, in file order.
    pub messages: Vec<Message>,
    /// `DRAFT <model|path>`: speculative-decoding draft model.
    pub draft: Option<String>,
    /// `DRAFT_MAX <n>`: max draft tokens (`--spec-draft-n-max`).
    pub draft_max: Option<u32>,
    /// `ADAPTER <path>`: LoRA adapter (llama.cpp `--lora`).
    pub adapter: Option<String>,
    /// `MMPROJ <model|path>`: vision projector (llama.cpp `--mmproj`).
    /// Omitted = auto-detect a same-repo mmproj blob at spawn time.
    pub mmproj: Option<String>,
    /// `KEEP_ALIVE <duration>`: e.g. `10m`, `1h`, `0`, `-1` (forever).
    pub keep_alive: Option<String>,
}

pub fn parse(path: &Path) -> Result<Modelfile> {
    let text = std::fs::read_to_string(path)?;
    parse_str(&text)
}

pub fn parse_str(text: &str) -> Result<Modelfile> {
    let mut m = Modelfile::default();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (verb, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        match verb.to_uppercase().as_str() {
            "FROM" => m.from = unquote(rest),
            "SYSTEM" => m.system = Some(take_block(rest, &mut lines)),
            "TEMPLATE" => m.template = Some(take_block(rest, &mut lines)),
            "LICENSE" => m.license = Some(take_block(rest, &mut lines)),
            "PARAMETER" => {
                let (k, v) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                m.parameters.insert(k.to_lowercase(), unquote(v.trim()));
            }
            "MESSAGE" => {
                // `MESSAGE <role> <content>`; role must be system|user|assistant.
                let (role, content) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                let role = role.to_lowercase();
                if !matches!(role.as_str(), "system" | "user" | "assistant") {
                    anyhow::bail!(
                        "MESSAGE role must be one of \"system\", \"user\", or \"assistant\" (got \"{role}\")"
                    );
                }
                m.messages.push(Message { role, content: take_block(content.trim(), &mut lines) });
            }
            "DRAFT" => m.draft = Some(unquote(rest)),
            "KEEP_ALIVE" => m.keep_alive = Some(unquote(rest)),
            "DRAFT_MAX" => {
                m.draft_max = Some(unquote(rest).parse().map_err(|_| {
                    anyhow::anyhow!("DRAFT_MAX must be a positive integer (got \"{rest}\")")
                })?);
            }
            "ADAPTER" => m.adapter = Some(unquote(rest)),
            "MMPROJ" => m.mmproj = Some(unquote(rest)),
            _ => {}
        }
    }
    if m.from.is_empty() {
        anyhow::bail!("Modelfile missing required FROM instruction");
    }
    Ok(m)
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    for q in ["\"\"\"", "'''", "\"", "'"] {
        if s.starts_with(q) && s.ends_with(q) && s.len() >= 2 * q.len() {
            return s[q.len()..s.len() - q.len()].to_string();
        }
    }
    s.to_string()
}

fn take_block(first: &str, lines: &mut std::iter::Peekable<std::str::Lines>) -> String {
    // `"""...` on one line, or block until closing `"""`.
    let t = first.trim();
    if (t.starts_with("\"\"\"") || t.starts_with("'''")) && t.len() > 3 {
        let q = &t[..3];
        if let Some(end) = t[3..].find(q) {
            return t[3..3 + end].to_string();
        }
        let mut acc = vec![t[3..].to_string()];
        while let Some(l) = lines.next() {
            if let Some(end) = l.find(q) {
                acc.push(l[..end].to_string());
                break;
            }
            acc.push(l.to_string());
        }
        return acc.join("\n");
    }
    unquote(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_full_modelfile() {
        let m = parse_str("FROM ./a.gguf\nPARAMETER temperature 0.2\nSYSTEM \"\"\"Be nice.\"\"\"\n").unwrap();
        assert_eq!(m.from, "./a.gguf");
        assert_eq!(m.parameters["temperature"], "0.2");
        assert_eq!(m.system.unwrap(), "Be nice.");
    }
    #[test]
    fn rejects_missing_from() {
        assert!(parse_str("PARAMETER temperature 1\n").is_err());
    }
    #[test]
    fn parses_message_directives() {
        let m = parse_str(
            "FROM ./a.gguf\nMESSAGE user Hello there\nMESSAGE assistant \"\"\"Hi!\nHow can I help?\"\"\"\n",
        )
        .unwrap();
        assert_eq!(m.messages.len(), 2);
        assert_eq!(m.messages[0].role, "user");
        assert_eq!(m.messages[0].content, "Hello there");
        assert_eq!(m.messages[1].role, "assistant");
        assert_eq!(m.messages[1].content, "Hi!\nHow can I help?");
    }
    #[test]
    fn rejects_invalid_message_role() {
        assert!(parse_str("FROM ./a.gguf\nMESSAGE robot hi\n").is_err());
    }
    #[test]
    fn parses_draft_and_adapter() {
        let m = parse_str("FROM ./a.gguf\nDRAFT ./tiny.gguf\nADAPTER ./lora.gguf\n").unwrap();
        assert_eq!(m.draft.unwrap(), "./tiny.gguf");
        assert_eq!(m.adapter.unwrap(), "./lora.gguf");
    }
    #[test]
    fn parses_mmproj() {
        let m = parse_str("FROM ./vl.gguf\nMMPROJ ./mmproj.gguf\n").unwrap();
        assert_eq!(m.mmproj.unwrap(), "./mmproj.gguf");
    }
    #[test]
    fn parses_draft_max() {
        let m = parse_str("FROM ./a.gguf\nDRAFT ./tiny.gguf\nDRAFT_MAX 5\n").unwrap();
        assert_eq!(m.draft_max, Some(5));
        assert!(parse_str("FROM ./a.gguf\nDRAFT_MAX lots\n").is_err());
    }
    #[test]
    fn parses_parameters() {
        let m = parse_str("FROM ./a.gguf\nPARAMETER num_ctx 8192\nPARAMETER num_batch 1024\n").unwrap();
        assert_eq!(m.parameters["num_ctx"], "8192");
        assert_eq!(m.parameters["num_batch"], "1024");
    }
}
