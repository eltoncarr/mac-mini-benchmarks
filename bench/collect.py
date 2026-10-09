#!/usr/bin/env python3
"""Aggregate a results/<run> directory into summary.json and summary.md."""
import json
import pathlib
import re
import statistics
import sys


def load(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def med(d, k):
    return ((d or {}).get(k) or {}).get("median")


def llama_bench(run, S, md):
    rows = []
    for f in sorted((run / "llama_bench").glob("*.json")):
        if f.stem == "4167":
            continue
        for r in load(f) or []:
            test = (f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r['n_gen']}")
            if r.get("n_depth"):
                test += f" @ d{r['n_depth']}"
            rows.append({"quant": f.stem, "test": test, "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts"),
                         "size_gib": (r.get("model_size") or 0) / 2**30, "build": r.get("build_commit"),
                         "spec": f.stem.endswith("-mtp")})
    S["llama_bench"] = rows
    if rows:
        md += ["## llama-bench (llama.cpp, Metal)", "", "| quant | test | t/s | ± |", "|---|---|---:|---:|"]
        md += [f"| {r['quant']} | {r['test']} | {fmt(r['avg_ts'], 2)} | {fmt(r['stddev_ts'], 2)} |" for r in rows]
        md += ["", f"build `{rows[0]['build']}`", ""]
    ref = run / "llama_bench" / "4167.md"
    if ref.exists():
        md += ["## Llama-2 7B reference (llama.cpp Discussion #4167 format)", "", ref.read_text().strip(), ""]


def mlx(run, S, md):
    out = {}
    for f in sorted((run / "mlx").glob("*.txt")):
        t = f.read_text()
        p = re.findall(r"prompt_tps=([\d.]+)", t)
        g = re.findall(r"generation_tps=([\d.]+)", t)
        m = re.findall(r"peak_memory=([\d.]+)", t)
        if p or g:
            out[f.stem] = {"prompt_tps": float(p[-1]) if p else None,
                           "generation_tps": float(g[-1]) if g else None,
                           "peak_memory_gb": float(m[-1]) if m else None}
    S["mlx"] = out
    if out:
        md += ["## MLX (mlx_lm.benchmark)", "", "| run | prompt t/s | gen t/s | peak mem GB |", "|---|---:|---:|---:|"]
        md += [f"| {k} | {fmt(v['prompt_tps'])} | {fmt(v['generation_tps'])} | {fmt(v['peak_memory_gb'])} |"
               for k, v in out.items()]
        md.append("")


def server(run, S, md):
    out = {}
    for f in sorted((run / "server").glob("*.json")):
        d = load(f)
        if d:
            out[f.stem] = d
    S["server"] = {k: {n: (v.get("summary") if isinstance(v, dict) and "summary" in v else v)
                       for n, v in d["scenarios"].items()} for k, d in out.items()}
    if not out:
        return
    md += ["## Server latency (OpenAI-compatible API, streaming)", "",
           "### Short chat (256 output tokens)", "",
           "| config | TTFT s | decode t/s | TPOT ms | ITL p50 / p99 ms | e2e s |", "|---|---:|---:|---:|---:|---:|"]
    for k, d in out.items():
        s = (d["scenarios"].get("short") or {}).get("summary")
        if s:
            md.append(f"| {k} | {fmt(med(s, 'ttft_s'), 3)} | {fmt(med(s, 'decode_tps'))} | {fmt(med(s, 'tpot_ms'))} | "
                      f"{fmt(med(s, 'itl_p50_ms'))} / {fmt(med(s, 'itl_p99_ms'))} | {fmt(med(s, 'e2e_s'), 2)} |")
    md += ["", "### Long context (prompt built from real source code, 64 output tokens)", "",
           "| config | prompt tokens | TTFT s | prefill t/s | decode t/s |", "|---|---:|---:|---:|---:|"]
    for k, d in out.items():
        for n, v in (d["scenarios"].get("longctx") or {}).items():
            s = v["summary"]
            md.append(f"| {k} | {n} | {fmt(med(s, 'ttft_s'), 2)} | {fmt(med(s, 'prefill_tps'))} | {fmt(med(s, 'decode_tps'))} |")
    for k, d in out.items():
        c = d["scenarios"].get("cache")
        if c:
            md += ["", f"Prompt cache ({k}, {c['prefix_tokens']} token prefix): cold TTFT {fmt(c['cold']['ttft_s'], 2)} s, "
                   f"warm TTFT {fmt(c['warm']['ttft_s'], 2)} s ({fmt(c['ttft_speedup'])}x)"]
    conc = [(k, d["scenarios"]["concurrency"]) for k, d in out.items() if d["scenarios"].get("concurrency")]
    if conc:
        md += ["", "### Concurrency (256 output tokens each)", "", "| config | parallel | aggregate t/s | median TTFT s |",
               "|---|---:|---:|---:|"]
        for k, c in conc:
            md += [f"| {k} | {lvl} | {fmt(v['aggregate_tps'])} | {fmt(med(v['summary'], 'ttft_s'), 2)} |" for lvl, v in c.items()]
    md.append("")


BUCKETS = [(0, 8192), (8192, 16384), (16384, 32768), (32768, 65536), (65536, 10**9)]


def agent_telemetry(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    rows = [r for r in rows if r.get("status") == 200 and r.get("context_tokens")]
    if not rows:
        return None
    out = {"calls": len(rows), "max_context": max(r["context_tokens"] for r in rows),
           "prompt_tokens_total": sum(r["context_tokens"] for r in rows),
           "generated_tokens_total": sum(r.get("predicted_n") or 0 for r in rows),
           "prefix_cache_hit_ratio": (sum(r.get("cache_n") or 0 for r in rows) /
                                      max(1, sum(r["context_tokens"] for r in rows))),
           "buckets": []}
    acc = [r for r in rows if r.get("draft_n")]
    if acc:
        out["mtp_acceptance"] = sum(r["draft_n_accepted"] or 0 for r in acc) / max(1, sum(r["draft_n"] for r in acc))
    for lo, hi in BUCKETS:
        b = [r for r in rows if lo <= r["context_tokens"] < hi]
        if not b:
            continue

        def m(k):
            v = [r[k] for r in b if r.get(k) is not None]
            return statistics.median(v) if v else None
        out["buckets"].append({"range": f"{lo // 1024}k-{hi // 1024 if hi < 10**9 else '∞'}k", "calls": len(b),
                               "ttft_s": m("ttft_s"), "prompt_tps": m("prompt_tps"),
                               "predicted_tps": m("predicted_tps"), "new_prompt_tokens": m("prompt_n")})
    return out


def swebench(run, S, md):
    d = run / "swebench"
    if not d.exists():
        return
    out = {}
    rep = load(d / "report.json")
    if rep:
        out["resolved"] = rep.get("resolved_instances")
        out["submitted"] = rep.get("submitted_instances") or rep.get("total_instances")
        out["completed"] = rep.get("completed_instances")
        out["errors"] = rep.get("error_instances")
    preds = load(d / "preds.json")
    if preds:
        out["predictions"] = len(preds)
        out["empty_patches"] = sum(1 for p in preds.values() if not (p.get("model_patch") or "").strip())
    if (d / "calls.jsonl").exists():
        out["telemetry"] = agent_telemetry(d / "calls.jsonl")
    meta = load(d / "meta.json") or {}
    out.update(meta)
    S["swebench"] = out
    md += ["## Agentic coding: SWE-bench (mini-swe-agent, bash-only)", ""]
    md.append(f"Subset **{meta.get('subset', '?')}**, step limit {meta.get('step_limit', '?')}, "
              f"thinking={'on' if meta.get('thinking') else 'off'}, wall time {fmt((meta.get('wall_s') or 0) / 3600, 2)} h")
    if out.get("resolved") is not None:
        md.append(f"\n**Resolved: {out['resolved']} / {out['submitted']}** "
                  f"({100 * out['resolved'] / max(1, out['submitted']):.1f}%), empty patches: {out.get('empty_patches')}")
    t = out.get("telemetry")
    if t:
        md += ["", f"{t['calls']} model calls, max context {t['max_context']} tokens, prefix-cache hit ratio "
               f"{100 * t['prefix_cache_hit_ratio']:.0f}%" + (f", MTP acceptance {100 * t['mtp_acceptance']:.0f}%" if t.get("mtp_acceptance") else ""),
               "", "| context | calls | median TTFT s | new prompt tokens | prefill t/s | decode t/s |",
               "|---|---:|---:|---:|---:|---:|"]
        md += [f"| {b['range']} | {b['calls']} | {fmt(b['ttft_s'], 2)} | {fmt(b['new_prompt_tokens'], 0)} | "
               f"{fmt(b['prompt_tps'])} | {fmt(b['predicted_tps'])} |" for b in t["buckets"]]
    md.append("")


def evalplus(run, S, md):
    out = {}
    for f in sorted((run / "evalplus").glob("*.log")):
        t = f.read_text()
        vals = re.findall(r"pass@1:\s*([\d.]+)", t)
        if vals:
            out[f.stem] = {"base_pass@1": float(vals[0]), "plus_pass@1": float(vals[1]) if len(vals) > 1 else None}
    S["evalplus"] = out
    if out:
        md += ["## Coding: EvalPlus (greedy, pass@1)", "", "| dataset | base | plus |", "|---|---:|---:|"]
        md += [f"| {k} | {fmt(100 * v['base_pass@1'])}% | {fmt(100 * v['plus_pass@1'] if v['plus_pass@1'] is not None else None)}% |"
               for k, v in out.items()]
        md.append("")


def power(run, S, md):
    out = {}
    for f in sorted((run / "power").glob("*.txt")):
        mw = [int(x) for x in re.findall(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW", f.read_text())]
        gpu = [int(x) for x in re.findall(r"GPU Power:\s*(\d+)\s*mW", f.read_text())]
        if mw:
            out[f.stem] = {"avg_w": statistics.mean(mw) / 1000, "max_w": max(mw) / 1000,
                           "gpu_avg_w": statistics.mean(gpu) / 1000 if gpu else None, "samples": len(mw)}
    S["power"] = out
    if out:
        md += ["## Power (powermetrics, CPU+GPU+ANE)", "", "| phase | avg W | max W | GPU avg W |", "|---|---:|---:|---:|"]
        md += [f"| {k} | {fmt(v['avg_w'])} | {fmt(v['max_w'])} | {fmt(v['gpu_avg_w'])} |" for k, v in out.items()]
        md.append("")


def main():
    run = pathlib.Path(sys.argv[1])
    env = load(run / "env.json") or {}
    S = {"run": run.name, "env": env}
    md = [f"# {env.get('model', 'Model')} on {env.get('machine', {}).get('model_name', 'Mac')} "
          f"({env.get('machine', {}).get('chip', '?')}, {env.get('machine', {}).get('memory_gb', '?')} GB)", "",
          f"Run `{run.name}` · macOS {env.get('machine', {}).get('macos', '?')} · "
          f"GPU cores {env.get('machine', {}).get('gpu_cores', '?')} · llama.cpp `{env.get('llama_cpp_commit', '?')}`", ""]
    for fn in (llama_bench, mlx, server, evalplus, swebench, power):
        try:
            fn(run, S, md)
        except Exception as e:  # keep collecting other sections
            md += [f"> {fn.__name__}: failed to parse ({e})", ""]
    (run / "summary.json").write_text(json.dumps(S, indent=2))
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print(run / "summary.md")


if __name__ == "__main__":
    main()
