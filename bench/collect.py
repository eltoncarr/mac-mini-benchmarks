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
        raw = ref.read_text().strip()
        md += ["## Llama-2 7B reference (llama.cpp Discussion #4167 format)", "", raw, ""]
        row = ref4167_row(raw, S.get("env") or {})
        if row:
            S["ref4167_row"] = row
            md += ["Summary row for the thread's table (bandwidth from Apple's spec for the chip):", "",
                   "| Chip | BW (GB/s) | GPU Cores | F16 PP | F16 TG | Q8_0 PP | Q8_0 TG | Q4_0 PP | Q4_0 TG | commit |",
                   "|---|---|---|---|---|---|---|---|---|---|", row, ""]


# Memory bandwidth (GB/s) by chip, from Apple's specs; used only for the #4167 row.
CHIP_BW = {"Apple M5 Pro": 307, "Apple M5 Max": 614, "Apple M5": 153, "Apple M4 Pro": 273, "Apple M4 Max": 546}


def ref4167_row(raw, env):
    """Build the one-line summary the #4167 thread uses from llama-bench's markdown output."""
    vals = {}
    for line in raw.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or not cells[-2] in ("pp512", "tg128"):
            continue
        model = cells[0].upper()
        kind = "F16" if "F16" in model else "Q8_0" if "Q8_0" in model else "Q4_0" if "Q4_0" in model else None
        if kind:
            vals[(kind, cells[-2][:2])] = cells[-1].split("±")[0].strip()
    build = re.search(r"build:\s*([0-9a-f]+)", raw)
    if not vals:
        return None
    m = env.get("machine", {})
    chip = m.get("chip", "?")
    cols = [vals.get((k, t), "—") for k in ("F16", "Q8_0", "Q4_0") for t in ("pp", "tg")]
    return (f"| {chip.replace('Apple ', '')} ({m.get('model_name', 'Mac')}, {m.get('memory_gb', '?')} GB) | "
            f"{CHIP_BW.get(chip, '?')} | {m.get('gpu_cores', '?')} | " + " | ".join(cols) +
            f" | {build.group(1) if build else '?'} |")


def vllm_bench(run, S, md):
    """`vllm bench serve` results (vllm/run.sh): <label>_in<N>_out<M>_c<C>.json."""
    rows = []
    for f in sorted((run / "vllm_bench").glob("*.json")):
        d = load(f)
        m = re.match(r"(.+)_in(\d+)_out(\d+)_c(\d+)$", f.stem)
        if not d or not m or not d.get("completed"):
            continue
        n_in, n_out, conc = int(m.group(2)), int(m.group(3)), int(m.group(4))
        ttft, tpot = d.get("median_ttft_ms"), d.get("median_tpot_ms")
        rows.append({"config": m.group(1), "n_in": n_in, "n_out": n_out, "concurrency": conc,
                     "completed": d.get("completed"), "ttft_ms": ttft, "ttft_p99_ms": d.get("p99_ttft_ms"),
                     "prefill_tps": n_in / ttft * 1000 if ttft else None,
                     "tpot_ms": tpot, "decode_tps": 1000 / tpot if tpot else None,
                     "itl_p99_ms": d.get("p99_itl_ms"), "e2el_ms": d.get("median_e2el_ms"),
                     "output_tps": d.get("output_throughput"), "total_tps": d.get("total_token_throughput")})
    rows.sort(key=lambda r: (r["config"], r["concurrency"], r["n_in"]))
    S["vllm_bench"] = rows
    # single-request sweep in llama-bench shape (pp = prefill of the whole prompt, tg = decode at that context)
    env = S.get("env") or {}
    label = next((x["label"] for x in env.get("models", [])), None)
    pp = env.get("perf_pp", 512)
    perf = [r for r in rows if r["config"].endswith("-perf")]
    S["vllm_perf"] = ([dict(model=label, n_depth=r["n_in"] - pp, n_prompt=r["n_in"], n_gen=None, avg_ts=r["prefill_tps"]) for r in perf] +
                      [dict(model=label, n_depth=r["n_in"] - pp, n_prompt=None, n_gen=r["n_out"], avg_ts=r["decode_tps"]) for r in perf])
    if not rows:
        return
    md += ["## vllm bench serve (random prompts, greedy, ignore_eos)", "",
           "Medians over the prompts. prefill t/s = prompt tokens / TTFT; decode t/s = 1000 / TPOT "
           "(per request). `-perf` is one request at a time; `-sweep` adds concurrency.", "",
           "| config | prompt | output | conc | TTFT ms | TTFT p99 | prefill t/s | TPOT ms | decode t/s | ITL p99 ms | out t/s total |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {r['config']} | {r['n_in']} | {r['n_out']} | {r['concurrency']} | {fmt(r['ttft_ms'], 0)} | "
           f"{fmt(r['ttft_p99_ms'], 0)} | {fmt(r['prefill_tps'])} | {fmt(r['tpot_ms'])} | {fmt(r['decode_tps'])} | "
           f"{fmt(r['itl_p99_ms'])} | {fmt(r['output_tps'])} |" for r in rows]
    md.append("")
    spec = {}
    for f in sorted((run / "server").glob("*.metrics.txt")):
        tot = {}
        for line in f.read_text().splitlines():
            m = re.match(r"^vllm:(spec_decode_num_(?:draft|accepted)_tokens|prefix_cache_(?:hits|queries))(?:_total)?(?:\{[^}]*\})?\s+([\d.e+]+)$", line)
            if m:
                tot[m.group(1)] = tot.get(m.group(1), 0) + float(m.group(2))
        if tot:
            spec[f.name.removesuffix(".metrics.txt")] = tot
    S["vllm_metrics"] = spec
    if spec:
        md += ["| server phase | drafted | accepted | acceptance | prefix-cache hit ratio |", "|---|---:|---:|---:|---:|"]
        for k, t in spec.items():
            dr, ac = t.get("spec_decode_num_draft_tokens"), t.get("spec_decode_num_accepted_tokens")
            q, h = t.get("prefix_cache_queries"), t.get("prefix_cache_hits")
            md.append(f"| {k} | {fmt(dr, 0)} | {fmt(ac, 0)} | {fmt(100 * ac / dr if dr else None)}{'%' if dr else ''} | "
                      f"{fmt(100 * h / q if q else None)}{'%' if q else ''} |")
        md.append("")


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
                md.append(f"| {label} | tg{b['response_size']} (after pp{b['prompt_size']}){at}{cc} | "
                          f"{fmt(m(b, 'tg_throughput'), 2)} | "
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
    # why each task ended (Submitted, LimitsExceeded, an exception name, ...)
    exits = {}
    for traj in sorted((d / "trajs").glob("*/*.traj.json")):
        st = ((load(traj) or {}).get("info") or {}).get("exit_status") or "unknown"
        exits[st] = exits.get(st, 0) + 1
    if exits:
        out["exit_statuses"] = exits
    # per task: resolved (True/False/None = not evaluated) and how it ended
    inst = {}
    for traj in sorted((d / "trajs").glob("*/*.traj.json")):
        inst[traj.parent.name] = {"resolved": None,
                                  "exit": ((load(traj) or {}).get("info") or {}).get("exit_status")}
    if rep:
        for iid in rep.get("resolved_ids") or []:
            inst.setdefault(iid, {})["resolved"] = True
        for key in ("unresolved_ids", "empty_patch_ids", "error_ids"):
            for iid in rep.get(key) or []:
                inst.setdefault(iid, {})["resolved"] = False
    out["instances"] = inst
    out.update(load(d / "meta.json") or {})
    return out


def swebench(run, S, md):
    root = run / "swebench"
    metas = sorted(root.rglob("meta.json")) if root.exists() else []
    if not metas:
        return
    # layout: swebench/<model>/<subset>/ (older runs: swebench/<model>/)
    res = {}
    for meta in metas:
        o = swebench_one(meta.parent)
        o.setdefault("model", meta.parent.relative_to(root).parts[0])
        res[str(meta.parent.relative_to(root))] = o
    S["swebench"] = res
    md += ["## Agentic coding: SWE-bench (mini-swe-agent, bash-only)", ""]
    md += swebench_table(res)
    for key, o in res.items():
        t = o.get("telemetry")
        if not t:
            continue
        md += ["", f"**{key}**: speed during the agent run by context size; prefix-cache hit ratio "
               f"{100 * t['prefix_cache_hit_ratio']:.0f}%"
               + (f", MTP acceptance {100 * t['mtp_acceptance']:.0f}%" if t.get("mtp_acceptance") else ""),
               "", "| context | calls | median TTFT s | new prompt tokens | prefill t/s | decode t/s |",
               "|---|---:|---:|---:|---:|---:|"]
        md += [f"| {b['range']} | {b['calls']} | {fmt(b['ttft_s'], 2)} | {fmt(b['new_prompt_tokens'], 0)} | "
               f"{fmt(b['prompt_tps'])} | {fmt(b['predicted_tps'])} |" for b in t["buckets"]]
    md.append("")


def swebench_table(res):
    md = ["| run | model | subset | resolved | empty patches | ended by | MTP | thinking | wall h | model calls | max context |",
          "|---|---|---|---:|---:|---|---|---|---:|---:|---:|"]
    for key, o in res.items():
        t = o.get("telemetry") or {}
        resolved = (f"{o['resolved']} / {o['submitted']} ({100 * o['resolved'] / max(1, o['submitted']):.1f}%)"
                    if o.get("resolved") is not None else "not evaluated")
        ended = ", ".join(f"{k} ×{v}" for k, v in (o.get("exit_statuses") or {}).items()) or "—"
        md.append(f"| {key} | {o.get('model')} | {o.get('subset')} | {resolved} | {o.get('empty_patches', '—')} | {ended} | "
                  f"{'on' if o.get('mtp') else 'off'} | {'on' if o.get('thinking') else 'off'} | "
                  f"{fmt((o.get('wall_s') or 0) / 3600, 2)} | {t.get('calls', '—')} | {t.get('max_context', '—')} |")
    # task-by-task, one column per model
    models = sorted({o["model"] for o in res.values()})
    tasks = {}
    for o in res.values():
        for iid, v in (o.get("instances") or {}).items():
            tasks.setdefault(iid, {})[o["model"]] = v
    if tasks:
        def cell(v):
            if not v:
                return ""
            mark = {True: "✅", False: "❌", None: "·"}[v.get("resolved")]
            return f"{mark} {v.get('exit') or ''}".strip()
        md += ["", "| task | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        md += [f"| {iid} | " + " | ".join(cell(per.get(m)) for m in models) + " |" for iid, per in sorted(tasks.items())]
        md += ["", "✅ resolved · ❌ not resolved · `·` not evaluated; the word is how the agent run ended."]
    return md


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

    perf = S.get("llama_bench", []) + S.get("vllm_perf", [])

    def lb(m, test, depth=0):
        for r in perf:
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

    def chat_tps(label):
        return ((S.get("server") or {}).get(label) or {}).get("decode_tps")

    sizes = {m["label"]: m["bytes"] / 2**30 for m in env.get("models", [])}
    # deepest context that every model was measured at (runs can use different sweeps)
    per_model = [{r["n_depth"] for r in perf if r["model"] == m} for m in models]
    common = set.intersection(*per_model) if all(per_model) else set()
    depth = max(common or {0})
    # llama.cpp runs label results llama.cpp-<model>[-mtp]; vLLM runs vllm-<model>[-ngram]
    info = {x["label"]: x for x in env.get("models", [])}
    srv = {m: (info.get(m) or {}).get("server_label", f"llama.cpp-{m}") for m in models}
    spec = {m: (info.get(m) or {}).get("spec", "mtp") for m in models}
    sp = "MTP" if set(spec.values()) == {"mtp"} else "spec"
    md = ["## Comparison", "",
          f"| model | size GiB | pp512 | tg128 | tg128 @ d{depth} | chat tg | chat tg {sp} | API tg | API tg {sp} | API TTFT ms | HumanEval+ | SWE-bench |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        ev = (S.get("evalplus", {}).get(m) or {}).get("humaneval") or {}
        sws = [o for o in (S.get("swebench") or {}).values() if o.get("model") == m and o.get("resolved") is not None]
        swr = ", ".join(f"{o['resolved']}/{o['submitted']} {o.get('subset', '')}".strip() for o in sws) or "—"
        md.append(f"| {m} | {fmt(sizes.get(m), 2)} | {fmt(lb(m, 'pp'))} | {fmt(lb(m, 'tg'))} | "
                  f"{fmt(lb(m, 'tg', depth)) if depth else '—'} | {fmt(chat_tps(srv[m]))} | "
                  f"{fmt(chat_tps(f'{srv[m]}-{spec[m]}'))} | {fmt(benchy_tg(srv[m]))} | "
                  f"{fmt(benchy_tg(f'{srv[m]}-{spec[m]}'))} | {fmt(ttft(srv[m]), 0)} | "
                  f"{fmt(100 * ev['plus_pass@1']) + '%' if ev.get('plus_pass@1') is not None else '—'} | {swr} |")
    md += ["", "pp/tg in tokens/s. llama-bench columns (pp512, tg128) are the engine alone. chat tg is the "
           "short-chat stream (256 tokens, thinking off) through llama-server; API columns are llama-benchy. "
           "MTP columns are empty when MTP did not run.", ""]
    if any(s != "mtp" for s in spec.values()):
        md += ["For vLLM models pp512/tg128 come from `vllm bench serve` (one request, through the server; the "
               "depth column is the prompt length minus 512), and the spec columns are n-gram speculation "
               "(vllm-metal has no MTP for hybrid Qwen3.8).", ""]
    md += [f"- `{m['label']}`: [{m['repo']}](https://huggingface.co/{m['repo']}) `{m['file']}`" for m in env.get("models", [])]
    return md + [""]


SECTIONS = (llama_bench, vllm_bench, benchy, server, mlx, evalplus, swebench, power)


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
            f"{engine(env)}", ""]


def engine(env):
    if env.get("engine"):
        v = env.get("engine_versions") or {}
        return f"{env['engine']} (vLLM {v.get('vllm', '?')}, MLX {v.get('mlx', '?')})"
    return f"llama.cpp `{env.get('llama_cpp_commit', '?')}`"


def compare(out, runs):
    merged, menv, notes = ({"llama_bench": [], "vllm_perf": [], "benchy": {}, "server": {}, "evalplus": {}, "swebench": {}},
                           {"models": []}, [])
    for run in runs:
        S, env, _ = gather(run)
        merged["llama_bench"] += S.get("llama_bench", [])
        merged["vllm_perf"] += S.get("vllm_perf", [])
        for k in ("benchy", "server", "evalplus", "swebench"):
            merged[k].update(S.get(k) or {})
        menv["models"] += env.get("models", [])
        menv.setdefault("first", env)
        notes.append(f"| `{run.name}` | {', '.join(m['label'] for m in env.get('models', [])) or '—'} | "
                     f"{env.get('suite', '?')} | {engine(env)} | {env.get('machine', {}).get('macos', '?')} |")
    commits = {n.split('|')[4].strip() for n in notes}
    suites = {n.split('|')[3].strip() for n in notes}
    md = header(menv["first"], ", ".join(r.name for r in runs)) + comparison(merged, menv)
    if merged["swebench"]:
        md += ["### SWE-bench", ""] + swebench_table(merged["swebench"]) + [""]
    md += ["### Runs", "", "| run | model | suite | engine | macOS |", "|---|---|---|---|---|"] + notes + [""]
    if len(commits) > 1:
        md += ["> Runs used different engines or llama.cpp commits; speed numbers may not be directly comparable.", ""]
    if len(suites) > 1:
        md += ["> Runs are different suites (e.g. quick vs standard); SWE-bench subsets and reps differ.", ""]
    pathlib.Path(out).write_text("\n".join(md) + "\n")
    print(out)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--compare":
        return compare(sys.argv[2], [pathlib.Path(r) for r in sys.argv[3:]])
    run = pathlib.Path(sys.argv[1])
    S, env, body = gather(run)
    md = header(env, run.name)
    probs = (run / "problems.log").read_text().strip().splitlines() if (run / "problems.log").exists() else []
    if probs:
        md += ["## Problems during this run", "", "Steps below failed or were skipped; their sections are missing. "
               "Details in `logs/`.", "", "```", *probs, "```", ""]
    S["problems"] = probs
    md += comparison(S, env) + body
    (run / "summary.json").write_text(json.dumps(S, indent=2))
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print(run / "summary.md")


if __name__ == "__main__":
    main()
