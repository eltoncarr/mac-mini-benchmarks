#!/usr/bin/env python3
"""Short-chat streaming latency for an OpenAI-compatible endpoint.

Complements llama-benchy (which covers pp/tg at depth, TTFT, concurrency and
prefix caching) with what it does not report: inter-token latency percentiles
(p50/p90/p99), TPOT and end-to-end latency for a typical short coding request
with a fixed 256-token answer.
"""
import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.request

PROMPT = (
    "Write a Python function `merge_intervals(intervals)` that merges overlapping "
    "closed intervals and returns them sorted. Include a docstring and three doctests."
)


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def chat(args, max_tokens):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,  # fixed output length (llama.cpp; ignored elsewhere)
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.perf_counter()
    arrivals, usage = [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    arrivals.append(time.perf_counter())
    t_end = time.perf_counter()
    if not arrivals:
        raise RuntimeError("no tokens received")
    n_out = (usage or {}).get("completion_tokens") or len(arrivals)
    decode_s = t_end - arrivals[0]
    itl = [b - a for a, b in zip(arrivals, arrivals[1:])] or [0.0]
    return {
        "completion_tokens": n_out,
        "ttft_s": arrivals[0] - t0,
        "e2e_s": t_end - t0,
        "tpot_ms": decode_s / max(1, n_out - 1) * 1000,
        "decode_tps": (n_out - 1) / decode_s if decode_s > 0 else None,
        "itl_p50_ms": pct(itl, 50) * 1000,
        "itl_p90_ms": pct(itl, 90) * 1000,
        "itl_p99_ms": pct(itl, 99) * 1000,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chat(args, 16)  # warmup
    runs = [chat(args, 256) for _ in range(args.reps)]
    summary = {k: statistics.median(r[k] for r in runs) for k in runs[0] if runs[0][k] is not None}
    print(f"[server_bench] {args.label}: decode {summary.get('decode_tps', 0):.1f} t/s, "
          f"ITL p50/p99 {summary['itl_p50_ms']:.1f}/{summary['itl_p99_ms']:.1f} ms", file=sys.stderr)
    out = {"label": args.label, "base_url": args.base_url, "summary": summary, "runs": runs,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
