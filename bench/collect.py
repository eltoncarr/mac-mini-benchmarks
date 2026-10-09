#!/usr/bin/env python3
"""Aggregate a results/<run> directory into summary.json and summary.md.

  collect.py results/<run>
  collect.py --compare out.md results/<run> results/<run> ...   # one row per model
"""
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


def llama_bench(run, S, md):
    rows = []
    for f in sorted((run / "llama_bench").glob("*.json")):
        if f.stem == "4167":
            continue
        for r in load(f) or []:
            test = (f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r['n_gen']}")
            if r.get("n_depth"):
                test += f" @ d{r['n_depth']}"
            rows.append({"model": f.stem, "n_depth": r.get("n_depth") or 0, "n_prompt": r.get("n_prompt"),
                         "n_gen": r.get("n_gen"), "test": test, "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts"),
                         "size_gib": (r.get("model_size") or 0) / 2**30, "build": r.get("build_commit")})
    S["llama_bench"] = rows
    if rows:
        md += ["## llama-bench (llama.cpp, Metal)", "", "| model | size GiB | test | t/s | ± |", "|---|---:|---|---:|---:|"]
        md += [f"| {r['model']} | {fmt(r['size_gib'], 2)} | {r['test']} | {fmt(r['avg_ts'], 2)} | {fmt(r['stddev_ts'], 2)} |"
               for r in rows]
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


def benchy(run, S, md):
    out = {}
    for f in sorted((run / "benchy").glob("*.json")):
        d = load(f)
        if d:
            out[f.stem] = d
    S["benchy"] = out
    if not out:
        return
    md += ["## llama-benchy (OpenAI API, llama-bench style)", "",
           "pp/tg in t/s (mean ± std); latencies in ms. TTFR = time to first response, "
           "est_ppt = estimated prompt processing time, e2e_ttft = end-to-end time to first token.", "",
           "| config | test | t/s | ± | peak t/s | TTFR | est_ppt | e2e_ttft |", "|---|---|---:|---:|---:|---:|---:|---:|"]

    def m(b, k):
        return (b.get(k) or {}).get("mean")

    def sd(b, k):
        return (b.get(k) or {}).get("std")
    for label, d in out.items():
        for b in d.get("benchmarks", []):
            at = f" @ d{b['context_size']}" if b.get("context_size") else ""
            cc = f" (c{b['concurrency']})" if b.get("concurrency", 1) > 1 else ""
            lat = f"{fmt(m(b, 'ttfr'), 0)} | {fmt(m(b, 'est_ppt'), 0)} | {fmt(m(b, 'e2e_ttft'), 0)}"
            if b.get("is_context_prefill_phase"):
                md.append(f"| {label} | ctx_pp{at}{cc} | {fmt(m(b, 'pp_throughput'), 2)} | {fmt(sd(b, 'pp_throughput'), 2)} | — | {lat} |")
                continue
            if m(b, "pp_throughput") is not None:
                md.append(f"| {label} | pp{b['prompt_size']}{at}{cc} | {fmt(m(b, 'pp_throughput'), 2)} | "
                          f"{fmt(sd(b, 'pp_throughput'), 2)} | — | {lat} |")
            if m(b, "tg_throughput") is not None:
                md.append(f"| {label} | tg{b['response_size']}{at}{cc} | {fmt(m(b, 'tg_throughput'), 2)} | "
                          f"{fmt(sd(b, 'tg_throughput'), 2)} | {fmt(m(b, 'peak_throughput'), 2)} | | | |")
    v = next(iter(out.values()))
    md += ["", f"llama-benchy {v.get('version')}, latency mode `{v.get('latency_mode')}`", ""]


def server(run, S, md):
    out = {}
    for f in sorted((run / "server").glob("*.json")):
        d = load(f)
        if d and "summary" in d:
            out[f.stem] = d["summary"]
    S["server"] = out
    if not out:
        return
    md += ["## Short-chat streaming latency (256 output tokens, thinking off)", "",
           "| config | TTFT s | decode t/s | TPOT ms | ITL p50 / p90 / p99 ms | e2e s |", "|---|---:|---:|---:|---:|---:|"]
    for k, s in out.items():
        md.append(f"| {k} | {fmt(s.get('ttft_s'), 3)} | {fmt(s.get('decode_tps'))} | {fmt(s.get('tpot_ms'))} | "
                  f"{fmt(s.get('itl_p50_ms'))} / {fmt(s.get('itl_p90_ms'))} / {fmt(s.get('itl_p99_ms'))} | {fmt(s.get('e2e_s'), 2)} |")
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


def swebench_one(d):
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
    out.update(load(d / "meta.json") or {})
    return out


def swebench(run, S, md):
    root = run / "swebench"
    dirs = sorted(p for p in root.glob("*") if p.is_dir() and (p / "meta.json").exists()) if root.exists() else []
    if not dirs:
        return
    res = {d.name: swebench_one(d) for d in dirs}
    S["swebench"] = res
    md += ["## Agentic coding: SWE-bench (mini-swe-agent, bash-only)", "",
           "| model | subset | resolved | empty patches | MTP | thinking | wall h | model calls | max context |",
           "|---|---|---:|---:|---|---|---:|---:|---:|"]
    for m, o in res.items():
        t = o.get("telemetry") or {}
        resolved = (f"{o['resolved']} / {o['submitted']} ({100 * o['resolved'] / max(1, o['submitted']):.1f}%)"
                    if o.get("resolved") is not None else "not evaluated")
        md.append(f"| {m} | {o.get('subset')} | {resolved} | {o.get('empty_patches', '—')} | "
                  f"{'on' if o.get('mtp') else 'off'} | {'on' if o.get('thinking') else 'off'} | "
                  f"{fmt((o.get('wall_s') or 0) / 3600, 2)} | {t.get('calls', '—')} | {t.get('max_context', '—')} |")
    for m, o in res.items():
        t = o.get("telemetry")
        if not t:
            continue
        md += ["", f"**{m}**: speed during the agent run by context size; prefix-cache hit ratio "
               f"{100 * t['prefix_cache_hit_ratio']:.0f}%"
               + (f", MTP acceptance {100 * t['mtp_acceptance']:.0f}%" if t.get("mtp_acceptance") else ""),
               "", "| context | calls | median TTFT s | new prompt tokens | prefill t/s | decode t/s |",
               "|---|---:|---:|---:|---:|---:|"]
        md += [f"| {b['range']} | {b['calls']} | {fmt(b['ttft_s'], 2)} | {fmt(b['new_prompt_tokens'], 0)} | "
               f"{fmt(b['prompt_tps'])} | {fmt(b['predicted_tps'])} |" for b in t["buckets"]]
    md.append("")


def evalplus(run, S, md):
    out = {}
    for f in sorted((run / "evalplus").glob("*/*.log")):
        vals = re.findall(r"pass@1:\s*([\d.]+)", f.read_text())
        if vals:
            out.setdefault(f.parent.name, {})[f.stem] = {
                "base_pass@1": float(vals[0]), "plus_pass@1": float(vals[1]) if len(vals) > 1 else None}
    S["evalplus"] = out
    if out:
        md += ["## Coding: EvalPlus (greedy, pass@1)", "", "| model | dataset | base | plus |", "|---|---|---:|---:|"]
        for m, dss in out.items():
            md += [f"| {m} | {k} | {fmt(100 * v['base_pass@1'])}% | "
                   f"{fmt(100 * v['plus_pass@1'] if v['plus_pass@1'] is not None else None)}% |" for k, v in dss.items()]
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


def comparison(S, env):
    """One row per model: the headline numbers side by side."""
    models = [m["label"] for m in env.get("models", [])] or sorted({r["model"] for r in S.get("llama_bench", [])})
    if not models:
        return []

    def lb(m, test, depth=0):
        for r in S.get("llama_bench", []):
            if r["model"] == m and r["n_depth"] == depth and ((test == "pp" and r["n_prompt"]) or (test == "tg" and r["n_gen"])):
                return r["avg_ts"]

    def benchy_tg(label):
        for b in (S.get("benchy", {}).get(label) or {}).get("benchmarks", []):
            if not b.get("context_size") and b.get("concurrency", 1) == 1 and b.get("tg_throughput"):
                return b["tg_throughput"]["mean"]

    def ttft(label):
        for b in (S.get("benchy", {}).get(label) or {}).get("benchmarks", []):
            if not b.get("context_size") and b.get("concurrency", 1) == 1 and b.get("e2e_ttft"):
                return b["e2e_ttft"]["mean"]

    sizes = {m["label"]: m["bytes"] / 2**30 for m in env.get("models", [])}
    # deepest context that every model was measured at (runs can use different sweeps)
    per_model = [{r["n_depth"] for r in S.get("llama_bench", []) if r["model"] == m} for m in models]
    common = set.intersection(*per_model) if all(per_model) else set()
    depth = max(common or {0})
    md = ["## Comparison", "",
          f"| model | size GiB | pp512 | tg128 | tg128 @ d{depth} | API tg | API tg MTP | API TTFT ms | HumanEval+ | SWE-bench |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        ev = (S.get("evalplus", {}).get(m) or {}).get("humaneval") or {}
        sw = (S.get("swebench") or {}).get(m) or {}
        swr = f"{sw['resolved']}/{sw['submitted']}" if sw.get("resolved") is not None else "—"
        md.append(f"| {m} | {fmt(sizes.get(m), 2)} | {fmt(lb(m, 'pp'))} | {fmt(lb(m, 'tg'))} | "
                  f"{fmt(lb(m, 'tg', depth)) if depth else '—'} | {fmt(benchy_tg(f'llama.cpp-{m}'))} | "
                  f"{fmt(benchy_tg(f'llama.cpp-{m}-mtp'))} | {fmt(ttft(f'llama.cpp-{m}'), 0)} | "
                  f"{fmt(100 * ev['plus_pass@1']) + '%' if ev.get('plus_pass@1') is not None else '—'} | {swr} |")
    md += ["", "pp/tg in tokens/s. llama-bench columns are the engine alone; API columns are llama-benchy "
           "through llama-server (MTP column empty when the model has no MTP head).", ""]
    md += [f"- `{m['label']}`: [{m['repo']}](https://huggingface.co/{m['repo']}) `{m['file']}`" for m in env.get("models", [])]
    return md + [""]


SECTIONS = (llama_bench, benchy, server, mlx, evalplus, swebench, power)


def gather(run):
    env = load(run / "env.json") or {}
    S = {"run": run.name, "env": env}
    md = []
    for fn in SECTIONS:
        try:
            fn(run, S, md)
        except Exception as e:  # keep collecting other sections
            md += [f"> {fn.__name__}: failed to parse ({e})", ""]
    return S, env, md


def header(env, run_name):
    m = env.get("machine", {})
    return [f"# {env.get('model', 'Model')} on {m.get('model_name', 'Mac')} ({m.get('chip', '?')}, {m.get('memory_gb', '?')} GB)",
            "", f"Run `{run_name}` · macOS {m.get('macos', '?')} · GPU cores {m.get('gpu_cores', '?')} · "
            f"llama.cpp `{env.get('llama_cpp_commit', '?')}`", ""]


def compare(out, runs):
    merged, menv, notes = {"llama_bench": [], "benchy": {}, "evalplus": {}, "swebench": {}}, {"models": []}, []
    for run in runs:
        S, env, _ = gather(run)
        merged["llama_bench"] += S.get("llama_bench", [])
        for k in ("benchy", "evalplus", "swebench"):
            merged[k].update(S.get(k) or {})
        menv["models"] += env.get("models", [])
        menv.setdefault("first", env)
        notes.append(f"| `{run.name}` | {', '.join(m['label'] for m in env.get('models', [])) or '—'} | "
                     f"{env.get('suite', '?')} | `{env.get('llama_cpp_commit', '?')}` | {env.get('machine', {}).get('macos', '?')} |")
    commits = {n.split('|')[4].strip() for n in notes}
    suites = {n.split('|')[3].strip() for n in notes}
    md = header(menv["first"], ", ".join(r.name for r in runs)) + comparison(merged, menv)
    md += ["### Runs", "", "| run | model | suite | llama.cpp | macOS |", "|---|---|---|---|---|"] + notes + [""]
    if len(commits) > 1:
        md += ["> Runs used different llama.cpp commits; speed numbers may not be directly comparable.", ""]
    if len(suites) > 1:
        md += ["> Runs are different suites (e.g. quick vs standard); SWE-bench subsets and reps differ.", ""]
    pathlib.Path(out).write_text("\n".join(md) + "\n")
    print(out)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--compare":
        return compare(sys.argv[2], [pathlib.Path(r) for r in sys.argv[3:]])
    run = pathlib.Path(sys.argv[1])
    S, env, body = gather(run)
    md = header(env, run.name) + comparison(S, env) + body
    (run / "summary.json").write_text(json.dumps(S, indent=2))
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print(run / "summary.md")


if __name__ == "__main__":
    main()
