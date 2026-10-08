#!/usr/bin/env python3
"""Race llmon vs official Ollama: same GGUF, same prompt, same token budget.
Metrics: TTFT (streaming) + wall time for N tokens. 1 warmup + 3 timed runs.
"""
import json
import statistics
import sys
import time
import urllib.request

PROMPT = "Explain gravity in one paragraph"
N = 60
TARGETS = {
    "llmon": "http://localhost:11437/api/generate",
    "ollama": "http://localhost:11434/api/generate",
}
# Same model identity per backend (same GGUF bytes underneath).
MODELS = {
    "llmon": "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    "ollama": "qwen-q40",
}


def run_once(url, model):
    body = json.dumps(
        {"model": model, "prompt": PROMPT, "stream": True,
         "options": {"num_predict": N, "temperature": 0, "num_thread": 2}}
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    chunks = 0
    text = []
    eval_count = eval_dur = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if first is None:
                first = time.monotonic() - t0
            text.append(c.get("response", ""))
            chunks += 1
            if c.get("done"):
                eval_count = c.get("eval_count")
                eval_dur = c.get("eval_duration")  # ns (ollama only)
                break
    total = time.monotonic() - t0
    return {"ttft": first or total, "total": total, "chars": sum(map(len, text)),
            "eval": (eval_count / (eval_dur / 1e9)) if eval_count and eval_dur else None}


def main():
    for name, url in TARGETS.items():
        model = MODELS[name]
        print(f"== {name} ({model}) warming up ==", flush=True)
        try:
            run_once(url, model)
        except Exception as e:
            print(f"{name} FAILED: {e}")
            continue
        t, s = [], []
        for i in range(3):
            r = run_once(url, model)
            t.append(r["ttft"])
            s.append(r["total"])
            extra = f" self={r['eval']:.1f} tok/s" if r["eval"] else ""
            print(f"  run{i+1}: TTFT={r['ttft']:.2f}s total={r['total']:.2f}s{extra}", flush=True)
        print(f"  => median TTFT={statistics.median(t):.2f}s "
              f"median total={statistics.median(s):.2f}s for ~{N} tokens\n", flush=True)


if __name__ == "__main__":
    sys.exit(main())
