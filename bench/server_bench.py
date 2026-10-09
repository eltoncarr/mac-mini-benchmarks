#!/usr/bin/env python3
"""Latency / throughput benchmark for an OpenAI-compatible chat endpoint.

Measures per request: TTFT, end-to-end latency, TPOT, inter-token latency
percentiles, decode tok/s, effective prefill tok/s (prompt_tokens / TTFT), and
the server-reported llama.cpp timings when present.

Scenarios
  short        short coding prompt, fixed 256 output tokens (chat-style TTFT/TG)
  longctx      prompts of N tokens built from real source code (prefill at depth)
  cache        same long prefix twice (prompt-cache cold vs warm TTFT)
  concurrency  L parallel requests (aggregate throughput)
"""
import argparse
import json
import pathlib
import random
import statistics
import sys
import threading
import time
import urllib.request

SHORT_PROMPT = (
    "Write a Python function `merge_intervals(intervals)` that merges overlapping "
    "closed intervals and returns them sorted. Include a docstring and three doctests."
)


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class Client:
    def __init__(self, args):
        self.base = args.base_url.rstrip("/")
        self.root = self.base[: -len("/v1")] if self.base.endswith("/v1") else self.base
        self.engine = args.engine
        self.model = args.model
        self.no_think = args.no_think

    def chat(self, prompt, max_tokens, fixed_len=False, cache_prompt=True):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        }
        if self.no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.engine == "llama":
            payload["cache_prompt"] = cache_prompt
            if fixed_len:
                payload["ignore_eos"] = True
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
        t0 = time.perf_counter()
        first, arrivals, usage, timings = None, [], None, None
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage") or usage
                timings = obj.get("timings") or timings
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if piece:
                        now = time.perf_counter()
                        first = first or now
                        arrivals.append(now)
        t_end = time.perf_counter()
        n_out = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n") or len(arrivals)
        n_in = (usage or {}).get("prompt_tokens")
        if n_in is None and timings:
            n_in = timings.get("prompt_n", 0) + timings.get("cache_n", 0)
        ttft = (first - t0) if first else None
        decode_s = (t_end - first) if first else None
        itl = [b - a for a, b in zip(arrivals, arrivals[1:])]
        return {
            "prompt_tokens": n_in,
            "completion_tokens": n_out,
            "ttft_s": ttft,
            "e2e_s": t_end - t0,
            "tpot_ms": (decode_s / (n_out - 1) * 1000) if first and n_out > 1 else None,
            "decode_tps": ((n_out - 1) / decode_s) if first and n_out > 1 and decode_s > 0 else None,
            "prefill_tps": (n_in / ttft) if n_in and ttft else None,
            "itl_p50_ms": (pct(itl, 50) or 0) * 1000 if itl else None,
            "itl_p90_ms": (pct(itl, 90) or 0) * 1000 if itl else None,
            "itl_p99_ms": (pct(itl, 99) or 0) * 1000 if itl else None,
            "server_timings": timings,
        }

    # --- prompt construction from a real code corpus --------------------------
    def tokenize(self, text):
        return http_json(self.root + "/tokenize", {"content": text})["tokens"]

    def detokenize(self, toks):
        return http_json(self.root + "/detokenize", {"tokens": toks})["content"]

    def text_of_tokens(self, corpus, n):
        if self.engine != "llama":
            return corpus[: int(n * 3.2)]  # no tokenizer endpoint; approximate
        chars = int(n * 4)
        while True:
            toks = self.tokenize(corpus[:chars])
            if len(toks) >= n or chars >= len(corpus):
                return self.detokenize(toks[:n])
            chars = int(chars * 1.3)


def load_corpus(path, limit_chars=3_000_000):
    root = pathlib.Path(path)
    parts, total = [], 0
    for f in sorted(root.rglob("*")):
        if f.suffix not in {".c", ".cpp", ".h", ".hpp", ".py", ".md"} or not f.is_file():
            continue
        try:
            t = f.read_text(errors="ignore")
        except OSError:
            continue
        parts.append(f"\n\n// ===== File: {f.relative_to(root)} =====\n{t}")
        total += len(parts[-1])
        if total > limit_chars:
            break
    if not parts:
        sys.exit(f"empty corpus at {path}")
    return "".join(parts)


def summarize(rows):
    keys = ["prompt_tokens", "completion_tokens", "ttft_s", "e2e_s", "tpot_ms", "decode_tps",
            "prefill_tps", "itl_p50_ms", "itl_p90_ms", "itl_p99_ms"]
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    return out


def server_ctx(client, default):
    if client.engine != "llama":
        return default
    try:
        props = http_json(client.root + "/props")
        return props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx") or default
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--engine", choices=["llama", "mlx", "other"], default="llama")
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", default="")
    ap.add_argument("--scenarios", default="short,longctx,cache")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sizes", default="1024 4096 16384 32768 65536")
    ap.add_argument("--concurrency", default="1 2 4")
    ap.add_argument("--corpus", required=True, help="directory of real source code for long prompts")
    ap.add_argument("--max-ctx", type=int, default=131072)
    ap.add_argument("--no-think", action="store_true", help="disable Qwen thinking via chat_template_kwargs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    c = Client(args)
    ctx = server_ctx(c, args.max_ctx)
    corpus = load_corpus(args.corpus)
    result = {"label": args.label, "engine": args.engine, "base_url": args.base_url,
              "n_ctx": ctx, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "scenarios": {}}
    scen = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    print(f"[server_bench] {args.label}: warmup", file=sys.stderr)
    c.chat("Say hi.", 16)

    if "short" in scen:
        rows = [c.chat(SHORT_PROMPT, 256, fixed_len=True) for _ in range(args.reps)]
        result["scenarios"]["short"] = {"summary": summarize(rows), "runs": rows}
        print(f"  short: {summarize(rows).get('decode_tps')}", file=sys.stderr)

    if "longctx" in scen:
        res = {}
        for n in [int(x) for x in args.sizes.split()]:
            if n + 512 > ctx:
                print(f"  longctx {n}: skipped (n_ctx={ctx})", file=sys.stderr)
                continue
            body = c.text_of_tokens(corpus, n)
            rows = []
            for _ in range(args.reps):
                prompt = (f"[session {random.getrandbits(64):x}]\nBelow is source code from a real repository.\n"
                          f"{body}\n\nIn three sentences, what is the overall purpose of this code?")
                rows.append(c.chat(prompt, 64, cache_prompt=False))
            res[str(n)] = {"summary": summarize(rows), "runs": rows}
            s = res[str(n)]["summary"]
            print(f"  longctx {n}: ttft={s['ttft_s']['median']:.2f}s prefill={s.get('prefill_tps', {}).get('median', 0):.0f} t/s",
                  file=sys.stderr)
        result["scenarios"]["longctx"] = res

    if "cache" in scen and c.engine == "llama":
        n = min(32768, ctx // 2)
        body = f"[session {random.getrandbits(64):x}]\n" + c.text_of_tokens(corpus, n)
        cold = c.chat(body + "\n\nList the three most important functions above.", 64)
        warm = c.chat(body + "\n\nWhich file above is the longest?", 64)
        result["scenarios"]["cache"] = {"prefix_tokens": n, "cold": cold, "warm": warm,
                                        "ttft_speedup": (cold["ttft_s"] / warm["ttft_s"]) if warm["ttft_s"] else None}
        print(f"  cache {n}: cold={cold['ttft_s']:.2f}s warm={warm['ttft_s']:.2f}s", file=sys.stderr)

    if "concurrency" in scen:
        res = {}
        for level in [int(x) for x in args.concurrency.split()]:
            rows, lock = [], threading.Lock()

            def worker():
                r = c.chat(SHORT_PROMPT, 256, fixed_len=True)
                with lock:
                    rows.append(r)
            t0 = time.perf_counter()
            ts = [threading.Thread(target=worker) for _ in range(level)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            wall = time.perf_counter() - t0
            total = sum(r["completion_tokens"] or 0 for r in rows)
            res[str(level)] = {"aggregate_tps": total / wall, "wall_s": wall, "summary": summarize(rows)}
            print(f"  concurrency {level}: aggregate {total / wall:.1f} t/s", file=sys.stderr)
        result["scenarios"]["concurrency"] = res

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
