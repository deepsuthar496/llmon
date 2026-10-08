//! Content-addressed model store.
//!
//! Layout (mirrors Ollama closely enough for familiarity, but simpler):
//! ```text
//! $MODELS/
//!   manifests/<registry>/<namespace>/<name>/<tag>.json
//!   blobs/sha256-<hex>            # GGUF bytes, mmap-friendly, shared
//!   tags/<name>:<tag>.json        # local alias -> manifest pointer
//! ```
//! Tags are tiny JSON files so `list/cp/rm` are atomic renames.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalManifest {
    pub name: String,
    pub tag: String,
    pub registry: String,
    pub blob_digest: String,
    pub size: u64,
    pub parameter_size: Option<String>,
    pub quantization: Option<String>,
    pub system: Option<String>,
    pub template: Option<String>,
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// `MESSAGE` few-shot turns from the Modelfile (ollama-compatible).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
    /// `DRAFT` model for speculative decoding (path or store alias).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft: Option<String>,
    /// `ADAPTER` LoRA path (llama.cpp `--lora`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// `MMPROJ` vision projector path (llama.cpp `--mmproj`).
    /// `None` = auto-detect a same-repo mmproj blob at spawn time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    /// `DRAFT_MAX` cap for speculative draft tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_max: Option<u32>,
    /// `KEEP_ALIVE` duration string (`10m`, `0`, `-1`); applied at spawn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// `PARAMETER` directives (num_ctx, num_batch, num_thread, ...) honored
    /// at spawn time; explicit `LLMON_*` env always wins.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters: HashMap<String, String>,
    pub created_at: String,
}

/// A stored few-shot turn (`MESSAGE <role> <content>`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Store {
    pub root: PathBuf,
}

impl Store {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }
    pub fn tags_dir(&self) -> PathBuf {
        self.root.join("tags")
    }
    pub fn manifests_dir(&self) -> PathBuf {
        self.root.join("manifests")
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [self.blobs_dir(), self.tags_dir(), self.manifests_dir()] {
            std::fs::create_dir_all(&d)
                .with_context(|| format!("create dir {}", d.display()))?;
        }
        Ok(())
    }

    fn tag_file(&self, name: &str, tag: &str) -> PathBuf {
        self.tags_dir().join(format!("{}-{}.json", safe(name), safe(tag)))
    }

    /// Split `name:tag` (default tag `latest`).
    pub fn split_name(s: &str) -> (String, String) {
        match s.rsplit_once(':') {
            Some((n, t)) if !t.contains('/') => (n.to_string(), t.to_string()),
            _ => (s.to_string(), "latest".to_string()),
        }
    }

    pub fn put_alias(&self, name: &str, tag: &str, m: &LocalManifest) -> Result<()> {
        self.ensure_dirs()?;
        let p = self.tag_file(name, tag);
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(m)?)?;
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }

    pub fn get(&self, model: &str) -> Result<LocalManifest> {
        let (name, tag) = Self::split_name(model);
        let p = self.tag_file(&name, &tag);
        let b = std::fs::read(&p)
            .with_context(|| format!("model '{model}' not found — run `pull {model}` first"))?;
        Ok(serde_json::from_slice(&b)?)
    }

    /// Exact lookup, else case-insensitive substring match over `name:tag`.
    /// Returns an error listing candidates when ambiguous.
    pub fn resolve(&self, model: &str) -> Result<LocalManifest> {
        if let Ok(m) = self.get(model) {
            return Ok(m);
        }
        let q = model.to_lowercase();
        let cands: Vec<LocalManifest> = self
            .list()
            .unwrap_or_default()
            .into_iter()
            .filter(|m| {
                format!("{}:{}", m.name, m.tag).to_lowercase().contains(&q)
                    || m.name.to_lowercase().contains(&q)
            })
            .collect();
        match cands.len() {
            1 => Ok(cands.into_iter().next().unwrap()),
            0 => anyhow::bail!(
                "model '{model}' not found locally — run `pull {model}` first (see `list` for available models)"
            ),
            _ => {
                let opts: Vec<String> =
                    cands.iter().map(|m| format!("{}:{}", m.name, m.tag)).collect();
                anyhow::bail!("'{model}' is ambiguous, pick one of: {}", opts.join(", "))
            }
        }
    }

    /// Filesystem path of the GGUF blob for a model (exact or fuzzy).
    pub fn gguf_path(&self, model: &str) -> Result<PathBuf> {
        let m = self.resolve(model)?;
        Ok(self.blob_path(&m.blob_digest))
    }

    /// Auto-detect a vision projector for `manifest`: a stored blob whose
    /// name contains "mmproj" and shares the owner/repo prefix. Returns the
    /// blob path of the first match (prefers Q8_0, then f16, then any).
    pub fn find_mmproj(&self, manifest: &LocalManifest) -> Option<PathBuf> {
        let repo_prefix = manifest.name.split('/').take(2).collect::<Vec<_>>().join("/");
        let all = self.list().unwrap_or_default();
        let mut cands: Vec<&LocalManifest> = all
            .iter()
            .filter(|m| {
                m.name.to_lowercase().contains("mmproj")
                    && (repo_prefix.is_empty() || m.name.starts_with(&repo_prefix))
            })
            .collect();
        cands.sort_by_key(|m| {
            let n = m.name.to_lowercase();
            if n.contains("q8_0") {
                0
            } else if n.contains("f16") {
                1
            } else {
                2
            }
        });
        // `all` outlives `cands`, so this borrow is safe.
        cands.first().map(|m| self.blob_path(&m.blob_digest))
    }

    #[allow(dead_code)]
    pub fn exists(&self, model: &str) -> bool {
        let (name, tag) = Self::split_name(model);
        self.tag_file(&name, &tag).exists()
    }

    pub fn list(&self) -> Result<Vec<LocalManifest>> {
        self.ensure_dirs()?;
        let mut out = vec![];
        for e in std::fs::read_dir(self.tags_dir())? {
            let e = e?;
            if e.path().extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            if let Ok(m) = serde_json::from_slice::<LocalManifest>(&std::fs::read(e.path())?) {
                out.push(m);
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name).then(a.tag.cmp(&b.tag)));
        Ok(out)
    }

    pub fn remove(&self, model: &str) -> Result<bool> {
        let (name, tag) = Self::split_name(model);
        let p = self.tag_file(&name, &tag);
        if !p.exists() {
            return Ok(false);
        }
        // Load manifest first so we can GC the blob if unreferenced.
        let m: LocalManifest = serde_json::from_slice(&std::fs::read(&p)?)?;
        std::fs::remove_file(&p)?;
        if self.list()?.iter().all(|x| x.blob_digest != m.blob_digest) {
            let bp = self.blob_path(&m.blob_digest);
            if bp.exists() {
                let _ = std::fs::remove_file(&bp);
            }
        }
        Ok(true)
    }

    pub fn copy(&self, src: &str, dst: &str) -> Result<()> {
        let m = self.get(src)?;
        let (name, tag) = Self::split_name(dst);
        let mut m2 = m;
        m2.name = name;
        m2.tag = tag;
        let (n, t) = (m2.name.clone(), m2.tag.clone());
        self.put_alias(&n, &t, &m2)
    }

    pub fn blob_path(&self, digest: &str) -> PathBuf {
        self.blobs_dir().join(digest.replace(':', "-"))
    }

    #[allow(dead_code)]
    pub fn blob_path_for_file(&self, filename: &str, digest_hint: Option<&str>) -> PathBuf {
        match digest_hint {
            Some(d) => self.blob_path(d),
            None => self.blobs_dir().join(safe(filename)),
        }
    }

    /// Import a local GGUF file (streaming sha256, no full read into RAM).
    pub fn import_file(&self, name: &str, tag: &str, src: &Path) -> Result<LocalManifest> {
        use std::io::{Read, Write};
        self.ensure_dirs()?;
        let mut hasher = Sha256::new();
        let mut size = 0u64;
        let f = std::fs::File::open(src).with_context(|| format!("open {}", src.display()))?;
        let mut reader = std::io::BufReader::new(f);
        let mut buf = vec![0u8; 1 << 20];
        // Hash first (streaming).
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            size += n as u64;
        }
        let digest = format!("sha256-{}", hex::encode(hasher.finalize()));
        let dst = self.blob_path(&digest);
        if !dst.exists() {
            // Copy via streaming; tmp + rename for atomicity.
            let tmp = dst.with_extension("tmp");
            let mut r = std::io::BufReader::new(std::fs::File::open(src)?);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            loop {
                let n = r.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                w.write_all(&buf[..n])?;
            }
            w.flush()?;
            std::fs::rename(&tmp, &dst)?;
        }
        let m = LocalManifest {
            name: name.to_string(),
            tag: tag.to_string(),
            registry: "local".to_string(),
            blob_digest: digest,
            size,
            parameter_size: None,
            quantization: guess_quant(src),
            system: None,
            template: None,
            license: None,
            verified: Some(true),
            messages: Vec::new(),
            draft: None,
            adapter: None,
            mmproj: None,
            draft_max: None,
            keep_alive: None,
            parameters: Default::default(),
            created_at: now_rfc3339(),
        };
        self.put_alias(name, tag, &m)?;
        Ok(m)
    }
}

fn safe(s: &str) -> String {
    s.replace(['/', '\\', ':', ' '], "-")
}

fn guess_quant(p: &Path) -> Option<String> {
    let n = p.file_name()?.to_string_lossy().to_uppercase();
    for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "F16", "F32", "IQ"] {
        if n.contains(q) {
            return Some(q.to_string());
        }
    }
    None
}

pub fn now_rfc3339() -> String {
    // No chrono dep on purpose (keep build light); epoch secs is enough.
    use std::time::{SystemTime, UNIX_EPOCH};
    let s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("epoch:{s}")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn splits_tags() {
        assert_eq!(Store::split_name("a"), ("a".into(), "latest".into()));
        assert_eq!(Store::split_name("a:q4"), ("a".into(), "q4".into()));
    }
}
