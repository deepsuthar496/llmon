#!/usr/bin/env python3
"""Side-by-side llmon vs Ollama: same GGUF, same prompts, same sampling.

Scenarios
  single   : 1 stream, /api/generate, temp 0, N tokens  (TTFT, wall, decode tok/s)
  parity   : /api/chat temp 0 — do both produce the same text? (template check)
  concur   : C simultaneous /api/chat requests          (aggregate tok/s, wall)
  multiturn: 4-turn conversation, TTFT per turn         (prompt-cache reuse)
  cold     : unload (keep_alive 0) then first request   (load + TTFT)

Usage: python3 scripts/bench.py [--llmon URL] [--ollama URL] [--threads T]
                                [--runs 3] [--tokens 128] [--concurrency 4]
                                [--only single,concur,...]
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPT = "Explain how gravity works in one detailed paragraph."
CONCUR_PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Explain the TCP three-way handshake step by step.",
    "List ten creative uses for a paperclip, with a sentence each.",
    "Describe the water cycle to a ten year old.",
    "Summarize the plot of Romeo and Juliet.",
    "Explain what a hash map is and how collisions are handled.",
    "Give a recipe for a simple tomato pasta.",
    "Why is the sky blue? Explain the physics.",
]
TURNS = [
    "Hi! I'm planning a trip to Japan. What cities should I visit?",
    "What about food? What should I try in each of those cities?",
    "How many days would you recommend for the whole trip?",
    "Great. Can you summarize the plan in a short bullet list?",
]


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream(url, body):
    """Return dict(ttft, total, text, final) from an Ollama NDJSON stream."""
    t0 = time.monotonic()
    first, text, final = None, [], {}
    with post(url, body) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if "error" in c:
                raise RuntimeError(c["error"])
            piece = c.get("response") or (c.get("message") or {}).get("content") or ""
            if piece and first is None:
                first = time.monotonic() - t0
            text.append(piece)
            if c.get("done"):
                final = c
                break
    total = time.monotonic() - t0
    return {"ttft": first or total, "total": total, "text": "".join(text), "final": final}


def rate(final):
    n, d = final.get("eval_count"), final.get("eval_duration")
    return n / (d / 1e9) if n and d else float("nan")


def med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


class Target:
    def __init__(self, name, base, model, opts):
        self.name, self.base, self.model, self.opts = name, base.rstrip("/"), model, opts

    def gen(self, prompt, n):
        return stream(f"{self.base}/api/generate", {
            "model": self.model, "prompt": prompt, "stream": True,
            "options": {**self.opts, "num_predict": n, "temperature": 0, "seed": 42}})

    def chat(self, messages, n, temp=0):
        return stream(f"{self.base}/api/chat", {
            "model": self.model, "messages": messages, "stream": True,
            "options": {**self.opts, "num_predict": n, "temperature": temp, "seed": 42}})

    def unload(self):
        with post(f"{self.base}/api/generate", {"model": self.model, "keep_alive": 0}) as r:
            r.read()
        time.sleep(1.0)


def sc_single(t, a):
    t.gen(PROMPT, 8)  # warmup / load
    rs = [t.gen(PROMPT, a.tokens) for _ in range(a.runs)]
    return {"ttft_s": med([r["ttft"] for r in rs]), "wall_s": med([r["total"] for r in rs]),
            "decode_tok_s": med([rate(r["final"]) for r in rs]),
            "tokens": rs[-1]["final"].get("eval_count")}


def sc_parity(t, a):
    r = t.chat([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], 40)
    return {"text": r["text"].strip()}


def sc_concur(t, a):
    c = a.concurrency
    prompts = (CONCUR_PROMPTS * 4)[:c]
    t.chat([{"role": "user", "content": "hi"}], 4)  # warm
    results = [None] * c

    def worker(i):
        results[i] = t.chat([{"role": "user", "content": prompts[i]}], a.tokens)

    t0 = time.monotonic()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.monotonic() - t0
    toks = sum(r["final"].get("eval_count") or 0 for r in results)
    return {"wall_s": wall, "total_tokens": toks, "aggregate_tok_s": toks / wall,
            "ttft_max_s": max(r["ttft"] for r in results)}


def sc_multiturn(t, a):
    msgs, ttfts = [], []
    for q in TURNS:
        msgs.append({"role": "user", "content": q})
        r = t.chat(msgs, 80)
        ttfts.append(round(r["ttft"], 3))
        msgs.append({"role": "assistant", "content": r["text"]})
    return {"ttft_per_turn_s": ttfts, "ttft_last_s": ttfts[-1]}


def sc_cold(t, a):
    t.unload()
    t0 = time.monotonic()
    r = t.gen(PROMPT, 16)
    return {"first_token_after_unload_s": r["ttft"],
            "load_s": (r["final"].get("load_duration") or 0) / 1e9,
            "wall_s": time.monotonic() - t0}


SCENARIOS = {"single": sc_single, "parity": sc_parity, "concur": sc_concur,
             "multiturn": sc_multiturn, "cold": sc_cold}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llmon", default="http://localhost:11437")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--llmon-model", default="Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf")
    p.add_argument("--ollama-model", default="qwen-q40")
    p.add_argument("--threads", type=int, default=0, help="num_thread sent to Ollama (0 = its default)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--only", default=",".join(SCENARIOS))
    p.add_argument("--json", help="write results here")
    a = p.parse_args()

    oopts = {"num_thread": a.threads} if a.threads else {}
    targets = [Target("llmon", a.llmon, a.llmon_model, {}),
               Target("ollama", a.ollama, a.ollama_model, oopts)]
    out = {}
    for sc in a.only.split(","):
        print(f"\n=== {sc} ===", flush=True)
        for t in targets:
            try:
                res = SCENARIOS[sc](t, a)
            except Exception as e:  # keep racing the other target
                res = {"error": str(e)}
            out.setdefault(sc, {})[t.name] = res
            pretty = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}
            print(f"  {t.name:7s} {json.dumps(pretty)}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
