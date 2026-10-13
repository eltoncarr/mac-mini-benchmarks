#!/usr/bin/env python3
"""Export published results as flat, machine-readable JSON and CSV.

One row per measurement (model, metric, context, prompt size, concurrency,
MTP, value, unit, tool), built from the runs' summary.json files, so the
data/ files always match results/.

  python bench/export.py --out data/qwen3.8-27b-mac-mini-m5-pro \\
      results/20261010-1516_Qwen3.8-27B_gsq-iq3s-mtp_final \\
      results/20261010-1905_Qwen3.8-27B_q8_0_final \\
      --ref4167 results/20261011-1307_ref4167
"""
import argparse
import csv
import json
import pathlib

FIELDS = ["model", "model_name", "metric", "context_tokens", "prompt_tokens", "concurrency", "mtp",
          "value", "unit", "tool", "note"]


def load(p):
    return json.loads(pathlib.Path(p).read_text())


def model_name(m):
    q = m.get("file", m["label"]).removesuffix(".gguf")
    return f"{q} ({m['repo'].split('/')[0]})"


def rows_for_run(run, rows):
    S = load(run / "summary.json")
    m = S["env"]["models"][0]
    label, name = m["label"], model_name(m)

    def add(metric, value, unit, tool, ctx=0, mtp=False, conc=1, prompt=None, note=None):
        rows.append({"model": label, "model_name": name, "metric": metric, "context_tokens": ctx,
                     "prompt_tokens": prompt, "concurrency": conc, "mtp": mtp, "value": round(value, 2),
                     "unit": unit, "tool": tool, "note": note})

    for b in S.get("llama_bench", []):
        add("prefill_tokens_per_s" if b["n_prompt"] else "generation_tokens_per_s", b["avg_ts"], "tokens/s",
            "llama-bench", ctx=b["n_depth"], prompt=b["n_prompt"] or None)
    def variant(name, prefix):
        """Suffix after the run's own label, e.g. "" / "-mtp" / "-prefix-cache". Model labels
        can themselves end in "-mtp" (gsq-iq3s-mtp), so never test the full name's ending."""
        return name[len(prefix):] if name.startswith(prefix) else None

    for lab, d in S.get("benchy", {}).items():
        mtp = variant(lab, f"llama.cpp-{label}") == "-mtp"
        kind = "prefix-cache" if "prefix-cache" in lab else "concurrency" if "concurrency" in lab else "api"
        tool = f"llama-benchy {d.get('version', '')} ({kind})".replace("  ", " ")
        for b in d.get("benchmarks", []):
            if b.get("is_context_prefill_phase"):
                continue
            ctx, conc, prompt = b["context_size"], b["concurrency"], b["prompt_size"]
            if b.get("pp_throughput"):
                add("prefill_tokens_per_s", b["pp_throughput"]["mean"], "tokens/s", tool, ctx, mtp, conc, prompt)
            if b.get("tg_throughput"):
                add("generation_tokens_per_s" if conc == 1 else "total_generation_tokens_per_s",
                    b["tg_throughput"]["mean"], "tokens/s", tool, ctx, mtp, conc, prompt)
            if b.get("e2e_ttft"):
                add("time_to_first_token_s", b["e2e_ttft"]["mean"] / 1000, "s", tool, ctx, mtp, conc, prompt,
                    "context served from prompt cache" if kind == "prefix-cache" else "uncached")
    for lab, s in S.get("server", {}).items():
        mtp = variant(lab, f"llama.cpp-{label}") == "-mtp"
        tool = "short-chat stream (256 tokens, code, thinking off)"
        add("generation_tokens_per_s", s["decode_tps"], "tokens/s", tool, mtp=mtp)
        add("time_to_first_token_s", s["ttft_s"], "s", tool, mtp=mtp)
    for ds, v in (S.get("evalplus", {}).get(label) or {}).items():
        tool = "EvalPlus (greedy, thinking off)"
        add(f"{ds}_pass_at_1", 100 * v["base_pass@1"], "%", tool)
        if v.get("plus_pass@1") is not None:
            add(f"{ds}_plus_pass_at_1", 100 * v["plus_pass@1"], "%", tool)
    for phase, p in S.get("power", {}).items():   # e.g. llama-bench-<label>, server-<label>, server-<label>-mtp
        stem, _, rest = phase.partition(f"-{label}")
        add("avg_power_w", p["avg_w"], "W", "powermetrics (CPU+GPU+ANE)", mtp=rest == "-mtp", note=stem)
    return S["env"], {**m, "name": name, "file_gib": round(m["bytes"] / 2**30, 2)}


def rows_for_ref4167(run, rows):
    S = load(run / "summary.json")
    row = [c.strip() for c in (S.get("ref4167_row") or "").strip("|").split("|")]
    if len(row) < 10:
        return
    commit = row[9]
    for kind, pp, tg in (("f16", row[3], row[4]), ("q8_0", row[5], row[6]), ("q4_0", row[7], row[8])):
        tool = f"llama-bench @ {commit} (llama.cpp Discussion #4167)"
        common = {"model": f"llama-2-7b-{kind}", "model_name": f"Llama-2 7B {kind.upper()} (ggml-org/Llama-2-7B-GGUF)",
                  "context_tokens": 0, "concurrency": 1, "mtp": False, "unit": "tokens/s", "tool": tool, "note": None}
        rows.append({**common, "metric": "prefill_tokens_per_s", "prompt_tokens": 512, "value": float(pp)})
        rows.append({**common, "metric": "generation_tokens_per_s", "prompt_tokens": None, "value": float(tg)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=pathlib.Path, help="final run folders (one per model)")
    ap.add_argument("--ref4167", type=pathlib.Path, help="ref4167 run folder")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--title", default="Qwen3.8-27B on Apple Mac mini M5 Pro (64 GB): llama.cpp benchmark, IQ3_S-mtp vs Q8_0")
    a = ap.parse_args()

    rows, models, env = [], [], {}
    for run in a.runs:
        env, m = rows_for_run(run, rows)
        models.append(m)
    if a.ref4167:
        rows_for_ref4167(a.ref4167, rows)
    mach = env.get("machine", {})
    doc = {
        "title": a.title,
        "source": "https://github.com/eltoncarr/mac-mini-benchmarks",
        "license": "CC-BY-4.0",
        "runs": [r.name for r in a.runs] + ([a.ref4167.name] if a.ref4167 else []),
        "hardware": {"device": mach.get("model_name"), "model_identifier": mach.get("model_id"), "chip": mach.get("chip"),
                     "cpu_cores": f"{mach.get('cpu_cores_perf')} performance + {mach.get('cpu_cores_eff')} efficiency",
                     "gpu_cores": int(mach.get("gpu_cores") or 0) or None, "memory_gb": mach.get("memory_gb"),
                     "gpu_wired_limit_mb": int(mach.get("gpu_wired_limit_mb") or 0) or None},
        "software": {"os": f"macOS {mach.get('macos')}", "engine": "llama.cpp (Metal)",
                     "llama_cpp_commit": env.get("llama_cpp_commit"), "llama_cpp_build": env.get("llama_cpp_describe"),
                     "kv_cache": env.get("kv_cache"), "server_ctx": env.get("server_ctx"), "mtp_args": env.get("mtp_args"),
                     "packages": env.get("packages")},
        "models": [{k: m[k] for k in ("label", "name", "repo", "file", "bytes", "file_gib")} for m in models],
        "fields": {"context_tokens": "tokens already in the context (llama-bench -d / llama-benchy --depth)",
                   "prompt_tokens": "new prompt tokens in the request (pp512 = 512)",
                   "concurrency": "parallel requests", "mtp": "multi-token-prediction speculative decoding on"},
        "results": rows,
    }
    # plain string paths: names like "qwen3.8-..." contain a dot that with_suffix would treat as an extension
    jpath, cpath = pathlib.Path(a.out + ".json"), pathlib.Path(a.out + ".csv")
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in FIELDS})
    print(f"{len(rows)} rows -> {jpath}, {cpath}")


if __name__ == "__main__":
    main()
