# llmon — run LLMs locally, fast

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-orange.svg)](https://www.rust-lang.org/)
[![Binary](https://img.shields.io/badge/binary-~7_MB-blue.svg)](#why-llmon)

A single ~7&nbsp;MB Rust binary that runs GGUF models locally with an
Ollama-compatible (`/api/*`) **and** OpenAI-compatible (`/v1/*`) API.
No Go runtime, no Python, no Docker required.

```bash
curl -fsSL https://raw.githubusercontent.com/deepsuthar496/llmon/main/install.sh \
  | LLMON_REPO=deepsuthar496/llmon bash
llmon pull Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
llmon serve &
llmon run Qwen "Hello!"
```

Point any OpenAI client at `http://localhost:11435/v1` — no code changes.

## Why llmon

| | llmon | Ollama 0.40 |
|---|---|---|
| 60 tokens, Qwen2.5-0.5B Q4_0 (unique prompts) | **2.9s** (~21 tok/s) | **2.8s** (~23 tok/s) |
| Same, repeated prompt | **~0.7s** | 2.4–2.6s |
| Daemon RSS | ~7 MB | ~41 MB |
| Install size | ~7 MB binary | GB-scale |

Same weights, same box (4 shared vCPUs, no GPU), medians of repeated runs —
see `scripts/race*.py` for the methodology. Takeaway: identical llama.cpp
kernels mean identical decode speed; llmon wins on everything *around*
decode (footprint, startup, zero-hop streaming, ngram speculative decoding
on repetitive/agent-loop prompts).

## Install

| Platform | Command |
|---|---|
| Linux / macOS | `curl -fsSL …/install.sh \| LLMON_REPO=deepsuthar496/llmon bash` |
| Windows (PowerShell) | `$env:LLMON_REPO="deepsuthar496/llmon"; irm …/install.ps1 \| iex` |
| Google Colab (T4) | open `colab-side-by-side.ipynb`, run all cells |
| Docker | `docker build -t llmon . && docker run -p 11435:11435 llmon` |
| From source | `cargo build --release --locked` → `target/release/llmon` |

Real (non-stub) inference needs the llama.cpp helpers on `PATH`:

```bash
./scripts/build-backend.sh            # CPU (~10 min, cmake + g++)
./scripts/build-backend.sh --cuda     # NVIDIA GPU (10–50x faster)
```

Without them llmon runs in clearly-labeled stub mode (`[llmon · stub]` prefix).

## Usage

```bash
llmon serve [--port 11435]            # daemon: /api/* + /v1/*
llmon run MODEL ["prompt..."]         # REPL when no prompt (/bye quits)
llmon pull OWNER/REPO/FILE.GGUF       # HF URL, hf:, registry: also accepted
llmon push MODEL OWNER/REPO           # upload to Hugging Face (HF_TOKEN)
llmon list | ps | show M | stop M     # manage local + resident models
llmon cp A B | rm A B...              # tag / delete (multi-rm supported)
llmon create NAME -f Modelfile [-q Q4_K_M]   # FROM/PARAMETER/SYSTEM/MESSAGE/…
llmon bench                           # throughput smoke test
```

`show --modelfile|--system|--template|--license`, `run --keepalive 5m`,
`--format json`, `--think [level]`, `--hidethinking`, per-model `PARAMETER
num_ctx/num_batch/num_thread/num_gpu` honored at spawn, `DRAFT`/`DRAFT_MAX`
speculative decoding, `ADAPTER` (LoRA), `MMPROJ` (vision, auto-paired).

Env: `LLMON_HOST`, `LLMON_PORT`, `LLMON_MODELS`, `LLMON_CTX`,
`LLMON_THREADS`, `LLMON_KEEP_ALIVE`, `LLMON_SKIP_VERIFY`, `HF_TOKEN`…

## API compatibility

Ollama-native: `/api/tags|ps|show|generate|chat|embed|pull|push|create|copy|delete|status|version`,
`POST /api/blobs/:digest`, `/tokenize`, `/detokenize`, `GET /v1/models/:model`,
`/v1/rerank`, `/v1/messages`, `/v1/responses`, `/v1/load_lora_adapter`,
`/tokenizer_info`, `/v1/audio/*`. OpenAI: `/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings` (all streaming-capable).

## Honest gaps (help wanted)

llmon is young. What's missing vs Ollama/vLLM:

- **No GPU machine has ever run the CUDA path** — biggest unknown; T4/4090
  numbers wanted (run the Colab notebook and report!).
- No model registry publishing beyond Hugging Face (`/api/push` is HF-only).
- No whisper/audio model support yet (endpoint proxies honestly 501 without
  an audio-capable model); no TTS, no realtime WS.
- No cloud features (accounts, web search, model recommendations with data).
- No prebuilt release binaries yet — installers build from source (~2 min).
- Benchmarks above are from one noisy shared box — reproduce before quoting.

If a gap blocks you, open an issue with the endpoint + a failing `curl`.

## Layout

```
src/{main,branding,config,store,registry,modelfile,template,engine,server,cli,repl}.rs
scripts/{serve,build-backend,race,race_clean}.sh|.py   install.sh / install.ps1
colab*.ipynb  Dockerfile  Modelfile.example
```

Rebrand in two places: `src/branding.rs` + `[[bin]] name` in `Cargo.toml`.

## License

MIT — see [LICENSE](LICENSE).
