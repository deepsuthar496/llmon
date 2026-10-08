//! Model downloads: Hugging Face GGUF files + Ollama-registry OCI blobs.
//! Concurrent, resumable (HTTP Range), progress-reporting.

use std::path::Path;

use anyhow::{Context, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

/// Download URL -> file, with optional streaming sha256 verification and progress callback.
/// Uses a single streaming GET (Range resume if `.part` exists).
/// When `verify` is true, computes SHA256 on the fly during download (single pass, zero re-read).
/// When `verify` is false, skips hashing CPU cycles while still asserting Content-Length.
pub async fn download_to(
    url: &str,
    dest: &Path,
    verify: bool,
    progress: Option<impl Fn(u64, Option<u64>) + Send + Sync + 'static>,
) -> Result<(u64, String)> {
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p)?;
    }
    let part = dest.with_extension("part");
    let resume_from = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let client = reqwest::Client::builder()
        .user_agent("llmon/0.1.0")
        .build()?;
    let mut req = client.get(url);
    if resume_from > 0 {
        req = req.header("Range", format!("bytes={resume_from}-"));
    }
    let resp = req.send().await.with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() && resp.status().as_u16() != 206 {
        anyhow::bail!("download failed: HTTP {} for {url}", resp.status());
    }
    let total = resp.content_length().map(|n| n + resume_from);

    let bar = ProgressBar::new(total.unwrap_or(0));
    bar.set_style(
        ProgressStyle::with_template("{msg} [{bar:40}] {bytes}/{total_bytes} {bytes_per_sec}")
            .unwrap()
            .progress_chars("=>-"),
    );
    bar.set_message("downloading");
    if resume_from > 0 {
        bar.set_position(resume_from);
    }

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(resume_from > 0)
        .write(true)
        .open(&part)
        .await?;
    let mut hasher = Sha256::new();
    // If resuming and verifying, hash existing prefix first (streaming, bounded RAM).
    if resume_from > 0 && verify {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let mut rf = tokio::fs::File::open(&part).await?;
        rf.seek(std::io::SeekFrom::Start(0)).await?;
        let mut buf = vec![0u8; 1 << 20];
        let mut left = resume_from;
        let cap = buf.len() as u64;
        while left > 0 {
            let n = rf.read(&mut buf[..left.min(cap) as usize]).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            left -= n as u64;
        }
    }
    use tokio::io::AsyncWriteExt;
    let mut stream = resp.bytes_stream();
    let mut downloaded = resume_from;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if verify {
            hasher.update(&chunk);
        }
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        bar.set_position(downloaded);
        if let Some(cb) = progress.as_ref() {
            cb(downloaded, total);
        }
    }
    // Assert size matches Content-Length if present (catches truncation without re-read)
    if let Some(expected) = total {
        if downloaded != expected {
            let _ = tokio::fs::remove_file(&part).await;
            anyhow::bail!("download truncated: expected {expected} bytes, received {downloaded} bytes");
        }
    }
    // fsync before rename closes crash-mid-write race
    file.sync_all().await?;
    drop(file);
    bar.finish_and_clear();
    tokio::fs::rename(&part, dest).await?;
    let digest = if verify {
        format!("sha256-{}", hex::encode(hasher.finalize()))
    } else {
        // Extract known sha256 from URL if present, otherwise use URL hash
        if let Some(pos) = url.find("sha256-") {
            let sub = &url[pos..];
            let d = sub.split(&['/', '?', '&', ':'][..]).next().unwrap_or(sub);
            d.to_string()
        } else if let Some(pos) = url.find("sha256:") {
            let sub = &url[pos + 7..];
            let d = sub.split(&['/', '?', '&'][..]).next().unwrap_or(sub);
            format!("sha256-{d}")
        } else {
            format!("sha256-unverified-{}", hex::encode(Sha256::digest(url.as_bytes())))
        }
    };
    Ok((downloaded, digest))
}

/// Resolve a `pull` argument to a direct download URL.
///
/// Accepted forms:
/// - `https://...gguf` direct URL
/// - `hf:<repo>/<file.gguf>` or `hf.co/<repo>/<file>`
/// - `<repo>/<file.gguf>` containing `.gguf` (assumed Hugging Face)
/// - `registry:<namespace>/<name>:<tag>` (Ollama registry, manifest lookup)
/// - bare `<name>` shorthand -> default HF demo repo file listing
pub async fn resolve_pull_url(spec: &str) -> Result<(String, String)> {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        let fname = spec.rsplit('/').next().unwrap_or("model.gguf").to_string();
        return Ok((spec.to_string(), fname));
    }
    let s = spec.strip_prefix("hf:").unwrap_or(spec);
    let s = s.strip_prefix("hf.co/").unwrap_or(s);
    if s.ends_with(".gguf") && s.contains('/') {
        // repo + file, or repo/subpath/file — split off last two? HF file may be nested.
        // Heuristic: first two segments are owner/model, rest is file path.
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() >= 3 {
            let repo = format!("{}/{}", parts[0], parts[1]);
            let file = parts[2..].join("/");
            let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
            return Ok((url, file.rsplit('/').next().unwrap_or(&file).to_string()));
        }
    }
    if let Some(rest) = spec.strip_prefix("registry:") {
        let (url, fname) = registry_blob_url(rest).await?;
        return Ok((url, fname));
    }
    // Bare name: query HF repo file list for first .gguf (small demo default).
    let repo = if spec.contains('/') {
        spec.to_string()
    } else {
        crate::branding::DEFAULT_HF_REPO.to_string()
    };
    let file = first_gguf_in_repo(&repo).await?;
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    Ok((url, file.rsplit('/').next().unwrap_or(&file).to_string()))
}

async fn first_gguf_in_repo(repo: &str) -> Result<String> {
    let api = format!("https://huggingface.co/api/models/{repo}");
    let v: serde_json::Value = reqwest::Client::builder()
        .user_agent("llmon/0.1.0")
        .build()?
        .get(&api)
        .send()
        .await?
        .json()
        .await?;
    if let Some(sibs) = v.get("siblings").and_then(|s| s.as_array()) {
        // Prefer smallest Q4 file for fast demo pulls.
        let mut cands: Vec<&str> = sibs
            .iter()
            .filter_map(|s| s.get("rfilename")?.as_str())
            .filter(|f| f.ends_with(".gguf"))
            .collect();
        cands.sort_by_key(|f| {
            if f.contains("Q4_K_M") {
                0
            } else if f.contains("Q4") {
                1
            } else {
                2
            }
        });
        if let Some(f) = cands.into_iter().next() {
            return Ok(f.to_string());
        }
    }
    anyhow::bail!("no .gguf found in HF repo {repo}")
}

async fn registry_blob_url(spec: &str) -> Result<(String, String)> {
    // spec: <namespace>/<name>:<tag>
    let (name, tag) = crate::store::Store::split_name(spec);
    let base = crate::branding::REGISTRY_BASE;
    let manifest_url = format!("{base}/v2/library/{name}/manifests/{tag}");
    let client = reqwest::Client::builder().user_agent("llmon/0.1.0").build()?;
    // Token (public library pulls are anonymous-friendly).
    let token_url = format!("{base}/v2/library/{name}/blobs/token?scope=pull");
    let _ = token_url;
    let resp = client
        .get(&manifest_url)
        .header("Accept", "application/vnd.docker.distribution.manifest.v2+json")
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("registry manifest lookup failed: HTTP {}", resp.status());
    }
    let v: serde_json::Value = resp.json().await?;
    // Find first GGUF-ish layer blob.
    if let Some(layers) = v.get("layers").and_then(|l| l.as_array()) {
        for l in layers {
            let digest = l.get("digest").and_then(|d| d.as_str()).unwrap_or("");
            let mt = l.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
            if mt.contains("model") || digest.len() > 7 {
                let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
                let url = format!("{base}/v2/library/{name}/blobs/sha256:{hex}");
                return Ok((url, format!("{name}-{tag}.gguf")));
            }
        }
    }
    anyhow::bail!("no model blob in registry manifest for {spec}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn test_single_pass_download_and_verify() {
        let payload = b"Hello, this is a test model blob for single-pass verification in llmon!";
        let expected_digest = format!("sha256-{}", hex::encode(Sha256::digest(payload)));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/test.gguf",
                axum::routing::get(|| async {
                    (
                        [
                            ("content-length", payload.len().to_string()),
                            ("content-type", "application/octet-stream".to_string()),
                        ],
                        payload.to_vec(),
                    )
                }),
            );
            let _ = axum::serve(listener, app).await;
        });

        let tmp_dir = std::env::temp_dir().join(format!("llmon-test-single-{}", std::process::id()));
        let dest = tmp_dir.join("model.gguf");
        let url = format!("http://127.0.0.1:{port}/test.gguf");

        let (size, digest) = download_to(&url, &dest, true, None::<fn(u64, Option<u64>)>)
            .await
            .unwrap();

        assert_eq!(size, payload.len() as u64);
        assert_eq!(digest, expected_digest);
        assert!(dest.exists());
        assert_eq!(std::fs::read(&dest).unwrap(), payload);
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[tokio::test]
    async fn test_skip_verify_flag() {
        let payload = b"Model blob with skipped verification";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/blob.gguf",
                axum::routing::get(|| async {
                    (
                        [("content-length", payload.len().to_string())],
                        payload.to_vec(),
                    )
                }),
            );
            let _ = axum::serve(listener, app).await;
        });

        let tmp_dir = std::env::temp_dir().join(format!("llmon-test-skip-{}", std::process::id()));
        let dest = tmp_dir.join("skip.gguf");
        let url = format!("http://127.0.0.1:{port}/blob.gguf");

        let (size, digest) = download_to(&url, &dest, false, None::<fn(u64, Option<u64>)>)
            .await
            .unwrap();

        assert_eq!(size, payload.len() as u64);
        assert!(digest.contains("sha256-"));
        assert!(dest.exists());
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}

/// Token for Hugging Face uploads (`HF_TOKEN` or `HF_HUB_TOKEN`).
pub fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HF_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.trim().is_empty())
}

/// Upload a local file to a Hugging Face repo (`owner/name`, file goes to
/// repo root unless `filename` contains directories).
///
/// Uses the Hub protocol: `preupload` (dedup check) -> PUT bytes to the
/// returned URL -> `commit` with an LFS pointer (GGUFs are always
/// LFS-tracked). Single streaming pass, progress-reporting.
/// Requires `HF_TOKEN`/`HF_HUB_TOKEN` with write access to the repo.
pub async fn push_to_hf(
    repo: &str,
    gguf_path: &Path,
    filename: &str,
    progress: Option<impl Fn(u64, Option<u64>) + Send + Sync + 'static>,
) -> Result<String> {
    let token = hf_token().ok_or_else(|| {
        anyhow::anyhow!("HF_TOKEN (or HF_HUB_TOKEN) is not set — create one at https://huggingface.co/settings/tokens with write access to '{repo}'")
    })?;
    let size = std::fs::metadata(gguf_path)
        .with_context(|| format!("open {}", gguf_path.display()))?
        .len();
    // Streaming sha256 of the file (bounded RAM) for the LFS pointer.
    let oid = file_sha256_hex(gguf_path).await?;

    let client = reqwest::Client::builder()
        .user_agent("llmon/0.1.0")
        .build()?;
    let auth = format!("Bearer {token}");

    // Best-effort repo creation (409 = already exists, fine).
    let _ = client
        .post("https://huggingface.co/api/repos/create")
        .header("Authorization", &auth)
        .json(&serde_json::json!({ "name": repo, "type": "model" }))
        .send()
        .await;

    // 1. preupload: Hub answers with a direct upload URL (or dedup hit).
    let pre: serde_json::Value = client
        .post(format!("https://huggingface.co/api/models/{repo}/preupload/main"))
        .header("Authorization", &auth)
        .json(&serde_json::json!({ "files": [{ "path": filename, "sample": "" }] }))
        .send()
        .await
        .with_context(|| format!("HF preupload for {repo}"))?
        .error_for_status()
        .with_context(|| format!("HF preupload rejected for '{repo}' (check token + write access)"))?
        .json()
        .await?;
    let files_empty = pre
        .get("files")
        .and_then(|f| f.as_array())
        .map(|a| a.is_empty())
        .unwrap_or(false);
    if files_empty {
        return Ok(format!("up to date: '{filename}' already in {repo}"));
    }
    let upload_url = pre
        .get("files")
        .and_then(|f| f.as_array())
        .and_then(|a| a.first())
        .and_then(|e| e.get("uploadUrl"))
        .and_then(|u| u.as_str())
        .ok_or_else(|| anyhow::anyhow!("HF preupload returned no upload URL for '{filename}'"))?
        .to_string();

    // 2. PUT the bytes (streamed from disk with progress).
    let bar = ProgressBar::new(size);
    bar.set_style(
        ProgressStyle::with_template("{msg} [{bar:40}] {bytes}/{total_bytes} {bytes_per_sec}")
            .unwrap()
            .progress_chars("=>-"),
    );
    bar.set_message("uploading");
    let bar2 = bar.clone();
    let stream = upload_stream(gguf_path, move |sent| {
        bar2.set_position(sent);
        if let Some(cb) = progress.as_ref() {
            cb(sent, Some(size));
        }
    });
    client
        .put(&upload_url)
        .header("Content-Type", "application/octet-stream")
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await
        .with_context(|| "HF blob upload failed")?
        .error_for_status()
        .context("HF blob upload rejected")?;
    bar.finish_and_clear();

    // 3. Commit the LFS pointer.
    let commit: serde_json::Value = client
        .post(format!("https://huggingface.co/api/models/{repo}/commit/main"))
        .header("Authorization", &auth)
        .json(&serde_json::json!({
            "operations": [{
                "key": "lfsFile",
                "value": { "path": filename, "algo": "sha256", "oid": oid, "size": size },
            }],
            "summary": format!("Upload {filename} with llmon"),
        }))
        .send()
        .await
        .context("HF commit failed")?
        .error_for_status()
        .context("HF commit rejected (check token write access)")?
        .json()
        .await
        .unwrap_or(serde_json::json!({}));
    let commit_oid = commit.get("commitOid").and_then(|o| o.as_str()).unwrap_or("committed");
    Ok(format!("pushed '{filename}' to {repo} (commit {commit_oid})"))
}

/// Streaming sha256 of a file without loading it into RAM.
async fn file_sha256_hex(path: &Path) -> Result<String> {
    use tokio::io::AsyncReadExt;
    let mut f = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// reqwest stream yielding file chunks + progress callbacks.
fn upload_stream(
    path: &Path,
    mut on_chunk: impl FnMut(u64) + Send + 'static,
) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send {
    use tokio::io::AsyncReadExt;
    let path = path.to_path_buf();
    async_stream::stream! {
        let mut f = match tokio::fs::File::open(&path).await {
            Ok(f) => f,
            Err(e) => { yield Err(e); return; }
        };
        let mut buf = vec![0u8; 1 << 20];
        let mut sent = 0u64;
        loop {
            match f.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    sent += n as u64;
                    on_chunk(sent);
                    yield Ok(bytes::Bytes::copy_from_slice(&buf[..n]));
                }
                Err(e) => { yield Err(e); break; }
            }
        }
    }
}
