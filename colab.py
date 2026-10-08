# llmon on Google Colab — one-click cell.
# Paste the whole block below into a single Colab code cell and run.
# It installs Rust (if needed), builds llmon (~1-2 min), pulls a small
# GGUF, starts the server in the background, and exposes OpenAI-compatible
# endpoints on localhost:11435. Use Colab's public-URL preview or port
# forwarding to reach it, or just call it from the same notebook via curl.

# --- cell start ---
# !set -e
# !if ! command -v cargo >/dev/null; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; fi
# !export PATH="$HOME/.cargo/bin:$PATH"
# !test -d /content/llmon || git clone --depth 1 https://github.com/deepsuthar496/llmon /content/llmon
# !cd /content/llmon && cargo build --release --locked
# !export LLMON_PORT=11435
# !nohup /content/llmon/target/release/llmon serve >/tmp/llmon.log 2>&1 &
# !sleep 3 && curl -s localhost:11435/health
# --- cell end ---
#
# Then, in the next cell:
#   !/content/llmon/target/release/llmon pull Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
#   !curl -s -H 'Content-Type: application/json' localhost:11435/api/generate \
#        -d '{"model":"qwen","prompt":"Hello!","stream":false}'

"""Programmatic Colab bootstrap (alternative to the shell cell above).

Usage in a Colab cell:
    !pip install -q requests  # only for the demo client part
    exec(open('/content/llmon/colab.py').read())  # after cloning
"""

import os
import subprocess
import sys
import time

REPO = os.environ.get("LLMON_REPO", "https://github.com/deepsuthar496/llmon")
PORT = os.environ.get("LLMON_PORT", "11435")


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    if subprocess.run("command -v cargo", shell=True).returncode != 0:
        sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
    if not os.path.exists("/content/llmon/Cargo.toml"):
        sh(f"git clone --depth 1 {REPO} /content/llmon")
    sh("cd /content/llmon && cargo build --release --locked")
    env = dict(os.environ, LLMON_PORT=PORT)
    subprocess.Popen(
        ["/content/llmon/target/release/llmon", "serve"],
        env=env,
        stdout=open("/tmp/llmon.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(30):
        time.sleep(1)
        r = subprocess.run(
            f"curl -sf localhost:{PORT}/health", shell=True, capture_output=True
        )
        if r.returncode == 0:
            print(r.stdout.decode())
            print(f"llmon is up on port {PORT}")
            return
    print(open("/tmp/llmon.log").read()[-4000:])
    sys.exit("server did not start — see /tmp/llmon.log")


if __name__ == "__main__":
    main()
