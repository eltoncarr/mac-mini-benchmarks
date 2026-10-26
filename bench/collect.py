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


# Memory bandwidth (GB/s) by chip, from Apple's and NVIDIA's specs; used only for the #4167 row.
CHIP_BW = {"Apple M5 Pro": 307, "Apple M5 Max": 614, "Apple M5": 153, "Apple M4 Pro": 273, "Apple M4 Max": 546,
           "NVIDIA GB10": 273}


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


def tbench_one(d):
    """One Harbor job: results/<run>/tbench/<model>/<scope>/job/<trial>/result.json."""
    trials = []
    for f in sorted((d / "job").glob("*/result.json")):
        r = load(f) or {}
        if "task_name" not in r:
            continue
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward", next(iter(rewards.values()), None))
        exc = (r.get("exception_info") or {}).get("exception_type")
        ex = r.get("agent_execution") or {}
        ar = r.get("agent_result") or {}
        trials.append({"task": r["task_name"].split("/")[-1], "reward": reward, "exception": exc,
                       "agent_s": seconds(ex.get("started_at"), ex.get("finished_at")),
                       "input_tokens": ar.get("n_input_tokens"), "output_tokens": ar.get("n_output_tokens")})
    out = dict(load(d / "meta.json") or {})
    out["trials"] = len(trials)
    out["passed"] = sum(1 for t in trials if (t["reward"] or 0) >= 1)
    out["score"] = statistics.mean([t["reward"] or 0 for t in trials]) if trials else None
    ends = {}
    for t in trials:
        k = "passed" if (t["reward"] or 0) >= 1 else (t["exception"] or "failed tests")
        ends[k] = ends.get(k, 0) + 1
    out["ended_by"] = ends
    out["agent_hours"] = sum(t["agent_s"] or 0 for t in trials) / 3600
    tasks = {}
    for t in trials:
        tasks.setdefault(t["task"], []).append(t)
    out["tasks"] = {k: {"passed": sum(1 for t in v if (t["reward"] or 0) >= 1), "trials": len(v),
                        "ended": sorted({t["exception"] or ("passed" if (t["reward"] or 0) >= 1 else "failed")
                                         for t in v})} for k, v in sorted(tasks.items())}
    if (d / "calls.jsonl").exists():
        out["telemetry"] = agent_telemetry(d / "calls.jsonl")
    return out


def seconds(a, b):
    from datetime import datetime
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except (TypeError, ValueError):
        return None


def tbench(run, S, md):
    root = run / "tbench"
    metas = sorted(root.rglob("meta.json")) if root.exists() else []
    if not metas:
        return
    res = {}
    for meta in metas:
        o = tbench_one(meta.parent)
        o.setdefault("model", meta.parent.relative_to(root).parts[0])
        res[str(meta.parent.relative_to(root))] = o
    S["tbench"] = res
    md += ["## Agentic coding: Terminal-Bench 2.0 (Harbor)", ""] + tbench_table(res)
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


def tbench_table(res):
    md = ["| run | agent | tasks passed | score | ended by | MTP | thinking | timeouts | images | at once | wall h | model calls | max context |",
          "|---|---|---:|---:|---|---|---|---:|---|---:|---:|---:|---:|"]
    for key, o in res.items():
        t = o.get("telemetry") or {}
        ended = ", ".join(f"{k} ×{v}" for k, v in sorted(o.get("ended_by", {}).items(), key=lambda kv: -kv[1])) or "—"
        images = (f"native {o.get('docker_arch') or ''}".strip() if o.get("native_build") else "published (x86_64)")
        md.append(f"| {key} | {o.get('agent', '?')} | {o['passed']} / {o['trials']} | "
                  f"{fmt(100 * o['score'] if o.get('score') is not None else None)}% | {ended} | "
                  f"{'on' if o.get('mtp') else 'off'} | {'on' if o.get('thinking') else 'off'} | "
                  f"×{o.get('timeout_multiplier', '?')} | {images} | {o.get('concurrent', '—')} | "
                  f"{fmt((o.get('wall_s') or 0) / 3600, 2)} | {t.get('calls', '—')} | {t.get('max_context', '—')} |")
    runs = [k for k in res]
    tasks = sorted({task for o in res.values() for task in o.get("tasks", {})})
    if tasks and len(tasks) <= 100:
        def cell(v):
            if not v:
                return ""
            mark = "✅" if v["passed"] == v["trials"] else ("❌" if not v["passed"] else f"{v['passed']}/{v['trials']}")
            why = ", ".join(e for e in v["ended"] if e not in ("passed", "failed"))
            return f"{mark} {why}".strip()
        md += ["", "| task | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
        md += [f"| {task} | " + " | ".join(cell(res[r].get("tasks", {}).get(task)) for r in runs) + " |" for task in tasks]
        md += ["", "✅ passed · ❌ failed (with the error type when the trial did not end normally, e.g. a timeout)."]
    oracle = [o for o in res.values() if o.get("agent") == "oracle"]
    for o in oracle:
        bad = [k for k, v in o.get("tasks", {}).items() if v["passed"] < v["trials"]]
        md += ["", f"Oracle check: the reference solutions pass {o['passed']} of {o['trials']} tasks on this machine's "
               f"images; these fail for every model here: {', '.join(bad) or 'none'}."]
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
        if not mw:  # Linux: nvidia-smi "timestamp, watts" lines (GPU board power only)
            gpu = [float(x) * 1000 for x in re.findall(r"^[^,\n]+,\s*([\d.]+)\s*$", f.read_text(), re.M)]
            mw = list(gpu)
        if mw:
            out[f.stem] = {"avg_w": statistics.mean(mw) / 1000, "max_w": max(mw) / 1000,
                           "gpu_avg_w": statistics.mean(gpu) / 1000 if gpu else None, "samples": len(mw)}
    S["power"] = out
    if out:
        src = "powermetrics, CPU+GPU+ANE" if (S.get("env") or {}).get("machine", {}).get("macos") else "nvidia-smi, GPU board"
        md += [f"## Power ({src})", "", "| phase | avg W | max W | GPU avg W |", "|---|---:|---:|---:|"]
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
          f"| model | size GiB | pp512 | tg128 | tg128 @ d{depth} | chat tg | chat tg {sp} | API tg | API tg {sp} | API TTFT ms | HumanEval+ | SWE-bench |"
          + (" Terminal-Bench 2.0 |" if S.get("tbench") else ""),
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if S.get("tbench") else "")]
    for m in models:
        ev = (S.get("evalplus", {}).get(m) or {}).get("humaneval") or {}
        sws = [o for o in (S.get("swebench") or {}).values() if o.get("model") == m and o.get("resolved") is not None]
        swr = ", ".join(f"{o['resolved']}/{o['submitted']} {o.get('subset', '')}".strip() for o in sws) or "—"
        tbs = [o for o in (S.get("tbench") or {}).values() if o.get("model") == m and o.get("trials")]
        tbr = ", ".join(f"{o['passed']}/{o['trials']} {o.get('scope', '')}".strip() for o in tbs) or "—"
        md.append(f"| {m} | {fmt(sizes.get(m), 2)} | {fmt(lb(m, 'pp'))} | {fmt(lb(m, 'tg'))} | "
                  f"{fmt(lb(m, 'tg', depth)) if depth else '—'} | {fmt(chat_tps(srv[m]))} | "
                  f"{fmt(chat_tps(f'{srv[m]}-{spec[m]}'))} | {fmt(benchy_tg(srv[m]))} | "
                  f"{fmt(benchy_tg(f'{srv[m]}-{spec[m]}'))} | {fmt(ttft(srv[m]), 0)} | "
                  f"{fmt(100 * ev['plus_pass@1']) + '%' if ev.get('plus_pass@1') is not None else '—'} | {swr} |"
                  + (f" {tbr} |" if S.get("tbench") else ""))
    md += ["", "pp/tg in tokens/s. llama-bench columns (pp512, tg128) are the engine alone. chat tg is the "
           "short-chat stream (256 tokens, thinking off) through llama-server; API columns are llama-benchy. "
           "MTP columns are empty when MTP did not run.", ""]
    if any(s != "mtp" for s in spec.values()):
        md += ["For vLLM models pp512/tg128 come from `vllm bench serve` (one request, through the server; the "
               "depth column is the prompt length minus 512), and the spec columns are n-gram speculation "
               "(vllm-metal has no MTP for hybrid Qwen3.8).", ""]
    md += [f"- `{m['label']}`: [{m['repo']}](https://huggingface.co/{m['repo']}) `{m['file']}`" for m in env.get("models", [])]
    return md + [""]


SECTIONS = (llama_bench, vllm_bench, benchy, server, mlx, evalplus, swebench, tbench, power)


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
    os_line = (f"macOS {m.get('macos', '?')} · GPU cores {m.get('gpu_cores', '?')}" if m.get("macos") or not m.get("os")
               else f"{m.get('os')} · driver {m.get('gpu_driver', '?')} · CUDA {m.get('cuda', '?')}")
    return [f"# {env.get('model', 'Model')} on {m.get('model_name') or 'Mac'} ({m.get('chip', '?')}, {m.get('memory_gb', '?')} GB)",
            "", f"Run `{run_name}` · {os_line} · {engine(env)}", ""]


def engine(env):
    if env.get("engine"):
        v = env.get("engine_versions") or {}
        return f"{env['engine']} (vLLM {v.get('vllm', '?')}, MLX {v.get('mlx', '?')})"
    return f"llama.cpp `{env.get('llama_cpp_commit', '?')}`"


def machine_name(env):
    m = env.get("machine", {})
    return m.get("model_name") or m.get("chip") or "?"


def relabel(S, env, tag):
    """Suffix every model label (and the result keys built from it) with ' @ tag'."""
    for x in env.get("models", []):
        old, srv = x["label"], x.get("server_label", f"llama.cpp-{x['label']}")
        new = f"{old} @ {tag}"
        x["label"], x["server_label"] = new, f"{srv} @ {tag}"
        for rows in (S.get("llama_bench", []), S.get("vllm_perf", [])):
            for r in rows:
                if r.get("model") == old:
                    r["model"] = new
        for k in ("benchy", "server"):
            S[k] = {(f"{srv} @ {tag}" + key[len(srv):] if key.startswith(srv) else key): v
                    for key, v in (S.get(k) or {}).items()}
        if old in (S.get("evalplus") or {}):
            S["evalplus"][new] = S["evalplus"].pop(old)
        for k in ("swebench", "tbench"):
            for o in (S.get(k) or {}).values():
                if o.get("model") == old:
                    o["model"] = new


def compare(out, runs):
    merged, menv, notes = ({"llama_bench": [], "vllm_perf": [], "benchy": {}, "server": {}, "evalplus": {}, "swebench": {}, "tbench": {}},
                           {"models": []}, [])
    gathered = [(run, *gather(run)[:2]) for run in runs]
    # Same model on several machines (e.g. Mac mini vs DGX Spark): keep the rows apart.
    multi = len({machine_name(env) for _, _, env in gathered}) > 1
    for run, S, env in gathered:
        if multi:
            relabel(S, env, machine_name(env))
        merged["llama_bench"] += S.get("llama_bench", [])
        merged["vllm_perf"] += S.get("vllm_perf", [])
        for k in ("benchy", "server", "evalplus"):
            merged[k].update(S.get(k) or {})
        mach = env.get("machine", {})
        for k in ("swebench", "tbench"):  # keyed <model>/<subset>: add the machine
            merged[k].update({(f"{key} @ {machine_name(env)}" if multi else key): v for key, v in (S.get(k) or {}).items()})
        menv["models"] += env.get("models", [])
        menv.setdefault("first", env)
        notes.append(f"| `{run.name}` | {', '.join(m['label'] for m in env.get('models', [])) or '—'} | "
                     f"{env.get('suite', '?')} | {engine(env)} | {mach.get('model_name') or mach.get('chip', '?')} | "
                     f"{mach.get('macos') or mach.get('os', '?')} |")
    commits = {n.split('|')[4].strip() for n in notes}
    suites = {n.split('|')[3].strip() for n in notes}
    md = header(menv["first"], ", ".join(r.name for r in runs))
    if multi:  # the header describes one machine; name them all
        names = list(dict.fromkeys(machine_name(env) for _, _, env in gathered))
        md = [f"# {menv['first'].get('model', 'Model')}: {' vs '.join(names)}", "",
              f"Runs {', '.join(f'`{r.name}`' for r in runs)}; machines and software in the Runs table below.", ""]
    md += comparison(merged, menv)
    if merged["swebench"]:
        md += ["### SWE-bench", ""] + swebench_table(merged["swebench"]) + [""]
    if merged["tbench"]:
        md += ["### Terminal-Bench 2.0", ""] + tbench_table(merged["tbench"]) + [""]
    md += ["### Runs", "", "| run | model | suite | engine | machine | OS |", "|---|---|---|---|---|---|"] + notes + [""]
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
