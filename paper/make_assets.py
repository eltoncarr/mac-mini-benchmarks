#!/usr/bin/env python3
"""Generate the paper's figures, tables and inline numbers from the released data.

Inputs (repository root): data/qwen3.8-27b-mac-mini-m5-pro.json and the final runs'
results/*/summary.json (for llama-bench standard deviations). Outputs:
  paper/figures/*.pdf       figures
  paper/generated/*.tex     tables and \\newcommand macros used in main.tex

  python paper/make_assets.py          # needs numpy and matplotlib
"""
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qwen3.8-27b-mac-mini-m5-pro.json"
RUNS = {"gsq-iq3s-mtp": ROOT / "results/20261010-1516_Qwen3.8-27B_gsq-iq3s-mtp_final",
        "q8_0": ROOT / "results/20261010-1905_Qwen3.8-27B_q8_0_final"}
FIG, GEN = ROOT / "paper/figures", ROOT / "paper/generated"
M = ["gsq-iq3s-mtp", "q8_0"]
NAME = {"gsq-iq3s-mtp": "IQ3\\_S-mtp", "q8_0": "Q8\\_0"}
PLOTNAME = {"gsq-iq3s-mtp": "IQ3_S-mtp (GSQ-RCO, 12 GB)", "q8_0": "Q8_0 (29 GB)"}
COLOR = {"gsq-iq3s-mtp": "#0072B2", "q8_0": "#D55E00"}   # Okabe-Ito, colour-blind safe
CTX = [0, 4096, 16384, 32768, 65536]
PEAK_BW = 307.0   # GB/s, Apple specification for M5 Pro
N_HE = 164        # HumanEval problems

D = json.loads(DATA.read_text())
R = D["results"]
BYTES = {m["label"]: m["bytes"] for m in D["models"]}


def one(model, metric, kind, **kw):
    """Exactly one matching row, or fail loudly (guards against filter mistakes)."""
    def ok(r):
        t = r["tool"]
        if kind == "bench" and t != "llama-bench":
            return False
        if kind == "chat" and not t.startswith("short-chat"):
            return False
        if kind == "evalplus" and not t.startswith("EvalPlus"):
            return False
        if kind in ("api", "prefix-cache", "concurrency") and not (t.startswith("llama-benchy") and f"({kind})" in t):
            return False
        if kind == "ref" and not t.startswith("llama-bench @"):
            return False
        return r["model"] == model and r["metric"] == metric and all(r.get(k) == v for k, v in kw.items())
    rows = [r for r in R if ok(r)]
    if len(rows) != 1:
        raise SystemExit(f"expected 1 row for {model} {metric} {kind} {kw}, got {len(rows)}")
    return rows[0]["value"]


def power(model, note, mtp):
    rows = [r["value"] for r in R if r["model"] == model and r["metric"] == "avg_power_w"
            and r.get("note") == note and r["mtp"] == mtp]
    assert len(rows) == 1, (model, note, mtp, rows)
    return rows[0]


def bench_sd(model, kind, depth):
    """llama-bench standard deviation from the run summary (not in the flat data file)."""
    S = json.loads((RUNS[model] / "summary.json").read_text())
    for b in S["llama_bench"]:
        if b["n_depth"] == depth and ((kind == "pp" and b["n_prompt"]) or (kind == "tg" and b["n_gen"])):
            return b["stddev_ts"]
    raise SystemExit(f"no llama-bench sd for {model} {kind} {depth}")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def linfit(x, y):
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2


# ---------------------------------------------------------------- derived values
V = {}
tg = {m: np.array([one(m, "generation_tokens_per_s", "bench", context_tokens=c) for c in CTX]) for m in M}
pp = {m: np.array([one(m, "prefill_tokens_per_s", "bench", context_tokens=c) for c in CTX]) for m in M}
x = np.array(CTX, float)
fit_tg = {m: linfit(x, 1000 / tg[m]) for m in M}
fit_pp = {m: linfit(x, 1000 / pp[m]) for m in M}
bw = {m: BYTES[m] * tg[m][0] / 1e9 for m in M}
api = {(m, mtp): np.array([one(m, "generation_tokens_per_s", "api", context_tokens=c, mtp=mtp, prompt_tokens=512) for c in CTX])
       for m in M for mtp in (False, True)}
chat = {(m, mtp): one(m, "generation_tokens_per_s", "chat", mtp=mtp) for m in M for mtp in (False, True)}
ttft_u = {m: np.array([one(m, "time_to_first_token_s", "api", context_tokens=c, mtp=False, prompt_tokens=512) for c in CTX[1:]]) for m in M}
ttft_c = {m: np.array([one(m, "time_to_first_token_s", "prefix-cache", context_tokens=c, prompt_tokens=512) for c in CTX[1:4]]) for m in M}
he = {(m, met): one(m, met, "evalplus") for m in M for met in ("humaneval_pass_at_1", "humaneval_plus_pass_at_1")}
he_k = {k: round(v / 100 * N_HE) for k, v in he.items()}
conc = {}
for m in M:
    for c in (1, 2, 4):
        metric = "generation_tokens_per_s" if c == 1 else "total_generation_tokens_per_s"
        conc[(m, c, 0)] = one(m, metric, "concurrency", context_tokens=0, prompt_tokens=512, concurrency=c)
        conc[(m, c, 4096)] = one(m, metric, "concurrency", context_tokens=4096, prompt_tokens=512, concurrency=c)
pw = {(m, mtp): power(m, "server", mtp) for m in M for mtp in (False, True)}
pw_bench = {m: power(m, "llama-bench", False) for m in M}
ref = {k: (one(f"llama-2-7b-{k}", "prefill_tokens_per_s", "ref"), one(f"llama-2-7b-{k}", "generation_tokens_per_s", "ref"))
       for k in ("f16", "q8_0", "q4_0")}
# MacBook Pro M5 Pro (20-core GPU) row of llama.cpp Discussion #4167, same commit c1d0e7a
MBP = {"f16": (1588.78, 21.55), "q8_0": (1553.86, 38.92), "q4_0": (1620.64, 66.33)}


def f(v, nd=1):
    return f"{v:.{nd}f}"


macros = {
    "IQsizeGB": f(BYTES["gsq-iq3s-mtp"] / 1e9, 2), "QEsizeGB": f(BYTES["q8_0"] / 1e9, 2),
    "sizeRatio": f(BYTES["q8_0"] / BYTES["gsq-iq3s-mtp"], 2),
    "IQtgZero": f(tg["gsq-iq3s-mtp"][0], 2), "QEtgZero": f(tg["q8_0"][0], 2),
    "IQtgLong": f(tg["gsq-iq3s-mtp"][-1], 2), "QEtgLong": f(tg["q8_0"][-1], 2),
    "tgRatioZero": f(tg["gsq-iq3s-mtp"][0] / tg["q8_0"][0], 2), "tgRatioLong": f(tg["gsq-iq3s-mtp"][-1] / tg["q8_0"][-1], 2),
    "IQppZero": f(pp["gsq-iq3s-mtp"][0], 1), "QEppZero": f(pp["q8_0"][0], 1),
    "IQppLong": f(pp["gsq-iq3s-mtp"][-1], 1), "QEppLong": f(pp["q8_0"][-1], 1),
    "IQbw": f(bw["gsq-iq3s-mtp"], 0), "QEbw": f(bw["q8_0"], 0),
    "IQbwPct": f(100 * bw["gsq-iq3s-mtp"] / PEAK_BW, 0), "QEbwPct": f(100 * bw["q8_0"] / PEAK_BW, 0),
    "IQfitA": f(fit_tg["gsq-iq3s-mtp"][0], 1), "QEfitA": f(fit_tg["q8_0"][0], 1),
    "IQfitB": f(fit_tg["gsq-iq3s-mtp"][1] * 1000, 3), "QEfitB": f(fit_tg["q8_0"][1] * 1000, 3),
    "IQfitR": f(fit_tg["gsq-iq3s-mtp"][2], 3), "QEfitR": f(fit_tg["q8_0"][2], 4),
    "IQppA": f(fit_pp["gsq-iq3s-mtp"][0], 2), "QEppA": f(fit_pp["q8_0"][0], 2),
    "IQppB": f(fit_pp["gsq-iq3s-mtp"][1] * 1000, 4), "QEppB": f(fit_pp["q8_0"][1] * 1000, 4),
    "IQppR": f(fit_pp["gsq-iq3s-mtp"][2], 3), "QEppR": f(fit_pp["q8_0"][2], 3),
    "IQchat": f(chat[("gsq-iq3s-mtp", False)], 1), "IQchatMTP": f(chat[("gsq-iq3s-mtp", True)], 1),
    "QEchat": f(chat[("q8_0", False)], 1), "QEchatMTP": f(chat[("q8_0", True)], 1),
    "IQchatGain": f(chat[("gsq-iq3s-mtp", True)] / chat[("gsq-iq3s-mtp", False)], 2),
    "QEchatGain": f(chat[("q8_0", True)] / chat[("q8_0", False)], 2),
    "chatRatioMTP": f(chat[("gsq-iq3s-mtp", True)] / chat[("q8_0", True)], 2),
    "IQmtpMin": f(min(api[("gsq-iq3s-mtp", True)] / api[("gsq-iq3s-mtp", False)]), 2),
    "IQmtpMax": f(max(api[("gsq-iq3s-mtp", True)] / api[("gsq-iq3s-mtp", False)]), 2),
    "QEmtpMin": f(min(api[("q8_0", True)] / api[("q8_0", False)]), 2),
    "QEmtpMax": f(max(api[("q8_0", True)] / api[("q8_0", False)]), 2),
    "mtpRatioMin": f(min(api[("gsq-iq3s-mtp", True)] / api[("q8_0", True)]), 2),
    "mtpRatioMax": f(max(api[("gsq-iq3s-mtp", True)] / api[("q8_0", True)]), 2),
    "IQhePlus": f(he[("gsq-iq3s-mtp", "humaneval_plus_pass_at_1")], 1), "QEhePlus": f(he[("q8_0", "humaneval_plus_pass_at_1")], 1),
    "IQheBase": f(he[("gsq-iq3s-mtp", "humaneval_pass_at_1")], 1), "QEheBase": f(he[("q8_0", "humaneval_pass_at_1")], 1),
    "IQjtok": f(pw[("gsq-iq3s-mtp", True)] / chat[("gsq-iq3s-mtp", True)], 2), "QEjtok": f(pw[("q8_0", True)] / chat[("q8_0", True)], 2),
    "IQjtokPlain": f(pw[("gsq-iq3s-mtp", False)] / chat[("gsq-iq3s-mtp", False)], 2),
    "QEjtokPlain": f(pw[("q8_0", False)] / chat[("q8_0", False)], 2),
    "cacheMax": f(max(ttft_u[m][2] / ttft_c[m][2] for m in M), 0),
    "IQttftLong": f(ttft_u["gsq-iq3s-mtp"][-1], 0), "QEttftLong": f(ttft_u["q8_0"][-1], 0),
}
# bandwidth floor (weights read once per token at the chip's peak) and the fitted model's ratio prediction
for m, key in (("gsq-iq3s-mtp", "IQ"), ("q8_0", "QE")):
    macros[f"{key}roofMs"] = f(BYTES[m] / 1e9 / PEAK_BW * 1000, 1)
    macros[f"{key}overMs"] = f(fit_tg[m][0] - BYTES[m] / 1e9 / PEAK_BW * 1000, 1)


def pred_ratio(ctx):
    (ai, bi, _), (aq, bq, _) = fit_tg["gsq-iq3s-mtp"], fit_tg["q8_0"]
    return (aq + bq * ctx) / (ai + bi * ctx)


macros["ratioPredLong"] = f(pred_ratio(65536), 2)
macros["ratioPredMax"] = f(pred_ratio(131072), 2)
macros["concEffIQ"] = f(conc[("gsq-iq3s-mtp", 4, 0)] / (4 * conc[("gsq-iq3s-mtp", 1, 0)]), 2)
macros["concEffQE"] = f(conc[("q8_0", 4, 0)] / (4 * conc[("q8_0", 1, 0)]), 2)
macros["IQbenchW"] = f(pw_bench["gsq-iq3s-mtp"], 1)
macros["QEbenchW"] = f(pw_bench["q8_0"], 1)
# HumanEval+ difference (unpaired Wald), the conservative bound without per-problem pairing
p1, p2 = he_k[("gsq-iq3s-mtp", "humaneval_plus_pass_at_1")] / N_HE, he_k[("q8_0", "humaneval_plus_pass_at_1")] / N_HE
se = math.sqrt(p1 * (1 - p1) / N_HE + p2 * (1 - p2) / N_HE)
macros["hePlusDiffHalfCI"] = f(100 * 1.96 * se, 1)
macros["hePlusK"] = str(he_k[("gsq-iq3s-mtp", "humaneval_plus_pass_at_1")])


# ---------------------------------------------------------------- tables
def row(cells):
    return " & ".join(cells) + r" \\"


def write(name, text):
    (GEN / name).write_text(text.strip() + "\n")


def table_sweep():
    lines = [r"\begin{tabular}{r rr rr rr rr}", r"\toprule",
             r" & \multicolumn{4}{c}{prefill, pp512 (tokens/s)} & \multicolumn{4}{c}{decode, tg128 (tokens/s)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"context & \multicolumn{2}{c}{IQ3\_S-mtp} & \multicolumn{2}{c}{Q8\_0} & \multicolumn{2}{c}{IQ3\_S-mtp} & \multicolumn{2}{c}{Q8\_0} \\",
             r" & mean & s.d. & mean & s.d. & mean & s.d. & mean & s.d. \\", r"\midrule"]
    for i, c in enumerate(CTX):
        cells = [f"{c:,}".replace(",", "\\,")]
        for kind, arr in (("pp", pp), ("tg", tg)):
            for m in M:
                cells += [f(arr[m][i], 2), f(bench_sd(m, kind, c), 2)]
        lines.append(row(cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_sweep.tex", "\n".join(lines))


def table_mtp():
    lines = [r"\begin{tabular}{r rrr rrr r}", r"\toprule",
             r" & \multicolumn{3}{c}{IQ3\_S-mtp} & \multicolumn{3}{c}{Q8\_0} & IQ3\_S/Q8\_0 \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"context & off & MTP & gain & off & MTP & gain & with MTP \\", r"\midrule"]
    for i, c in enumerate(CTX):
        a0, a1 = api[("gsq-iq3s-mtp", False)][i], api[("gsq-iq3s-mtp", True)][i]
        b0, b1 = api[("q8_0", False)][i], api[("q8_0", True)][i]
        lines.append(row([f"{c:,}".replace(",", "\\,"), f(a0, 2), f(a1, 2), f"{a1 / a0:.2f}$\\times$",
                          f(b0, 2), f(b1, 2), f"{b1 / b0:.2f}$\\times$", f"{a1 / b1:.2f}$\\times$"]))
    lines.append(r"\midrule")
    a0, a1, b0, b1 = chat[("gsq-iq3s-mtp", False)], chat[("gsq-iq3s-mtp", True)], chat[("q8_0", False)], chat[("q8_0", True)]
    lines.append(row(["code$^\\dagger$", f(a0, 2), f(a1, 2), f"{a1 / a0:.2f}$\\times$", f(b0, 2), f(b1, 2),
                      f"{b1 / b0:.2f}$\\times$", f"{a1 / b1:.2f}$\\times$"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_mtp.tex", "\n".join(lines))


def table_quality():
    lines = [r"\begin{tabular}{l rr rr}", r"\toprule",
             r" & \multicolumn{2}{c}{HumanEval (base tests)} & \multicolumn{2}{c}{HumanEval+ (base + extra)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"model & pass@1 & 95\% CI & pass@1 & 95\% CI \\", r"\midrule"]
    for m in M:
        cells = [NAME[m]]
        for met in ("humaneval_pass_at_1", "humaneval_plus_pass_at_1"):
            k = he_k[(m, met)]
            lo, hi = wilson(k, N_HE)
            cells += [f"{100 * k / N_HE:.1f}\\% ({k}/{N_HE})", f"[{100 * lo:.1f}, {100 * hi:.1f}]"]
        lines.append(row(cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_quality.tex", "\n".join(lines))


def table_serving():
    lines = [r"\begin{tabular}{l rr}", r"\toprule", r"measure & IQ3\_S-mtp & Q8\_0 \\", r"\midrule"]
    for c in (1, 2, 4):
        lines.append(row([f"total decode, {c} parallel request{'s' if c > 1 else ''}, 0 context (tokens/s)",
                          f(conc[("gsq-iq3s-mtp", c, 0)], 1), f(conc[("q8_0", c, 0)], 1)]))
    lines.append(row(["scaling efficiency at 4 requests, 0 context",
                      f(conc[("gsq-iq3s-mtp", 4, 0)] / (4 * conc[("gsq-iq3s-mtp", 1, 0)]), 2),
                      f(conc[("q8_0", 4, 0)] / (4 * conc[("q8_0", 1, 0)]), 2)]))
    lines.append(row(["total decode, 4 parallel requests, 4\\,096 context (tokens/s)",
                      f(conc[("gsq-iq3s-mtp", 4, 4096)], 1), f(conc[("q8_0", 4, 4096)], 1)]))
    lines.append(r"\midrule")
    lines.append(row(["average power, llama-bench (W)", f(pw_bench["gsq-iq3s-mtp"], 1), f(pw_bench["q8_0"], 1)]))
    for mtp in (False, True):
        lines.append(row([f"average power, server, MTP {'on' if mtp else 'off'} (W)", f(pw[("gsq-iq3s-mtp", mtp)], 1), f(pw[("q8_0", mtp)], 1)]))
    for mtp in (False, True):
        lines.append(row([f"energy per generated token, MTP {'on' if mtp else 'off'} (J)$^\\ddagger$",
                          f(pw[("gsq-iq3s-mtp", mtp)] / chat[("gsq-iq3s-mtp", mtp)], 2),
                          f(pw[("q8_0", mtp)] / chat[("q8_0", mtp)], 2)]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_serving.tex", "\n".join(lines))


def table_ref():
    lines = [r"\begin{tabular}{l rr rr rr}", r"\toprule",
             r" & \multicolumn{2}{c}{F16} & \multicolumn{2}{c}{Q8\_0} & \multicolumn{2}{c}{Q4\_0} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"system & pp512 & tg128 & pp512 & tg128 & pp512 & tg128 \\", r"\midrule"]
    lines.append(row(["Mac mini, M5 Pro (this work)"] + [f(v, 2) for k in ("f16", "q8_0", "q4_0") for v in ref[k]]))
    lines.append(row(["MacBook Pro, M5 Pro (Discussion \\#4167)"] + [f(v, 2) for k in ("f16", "q8_0", "q4_0") for v in MBP[k]]))
    lines.append(row(["ratio"] + [f"{a / b:.3f}" for k in ("f16", "q8_0", "q4_0") for a, b in zip(ref[k], MBP[k])]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_ref.tex", "\n".join(lines))


# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
XT = [0, 16384, 32768, 65536]
XL = ["0", "16k", "32k", "64k"]


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / name, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def fig_decode_latency():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    xx = np.linspace(0, 65536, 50)
    for m in M:
        a, b, r2 = fit_tg[m]
        ax.plot(x, 1000 / tg[m], "o", color=COLOR[m], ms=4, label=PLOTNAME[m])
        ax.plot(xx, a + b * xx, "-", color=COLOR[m], lw=1, alpha=0.7)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("decode time per token (ms)")
    ax.set_xticks(XT, XL)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "decode_latency.pdf")


def fig_throughput():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for m in M:
        ax.plot(x, api[(m, False)], "o-", color=COLOR[m], ms=3.5, lw=1, label=f"{PLOTNAME[m].split(' (')[0]}, MTP off")
        ax.plot(x, api[(m, True)], "s--", color=COLOR[m], ms=3.5, lw=1, label=f"{PLOTNAME[m].split(' (')[0]}, MTP on")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("decode throughput (tokens/s)")
    ax.set_xticks(XT, XL)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, ncol=2, loc="lower left", columnspacing=0.8)
    save(fig, "throughput_mtp.pdf")


def fig_prefill():
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for m in M:
        ax.plot(x, pp[m], "o-", color=COLOR[m], ms=3.5, lw=1, label=PLOTNAME[m])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("prefill throughput (tokens/s)")
    ax.set_xticks(XT, XL)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "prefill.pdf")


def fig_ttft():
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    xc = np.array(CTX[1:], float)
    for m in M:
        ax.plot(xc, ttft_u[m], "o-", color=COLOR[m], ms=3.5, lw=1, label=f"{PLOTNAME[m].split(' (')[0]}, uncached")
        ax.plot(xc[:3], ttft_c[m], "s--", color=COLOR[m], ms=3.5, lw=1, label=f"{PLOTNAME[m].split(' (')[0]}, prefix cached")
    ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("time to first token (s)")
    ax.set_xticks(XT[1:], XL[1:])
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    save(fig, "ttft.pdf")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    table_sweep()
    table_mtp()
    table_quality()
    table_serving()
    table_ref()
    fig_decode_latency()
    fig_throughput()
    fig_prefill()
    fig_ttft()
    write("numbers.tex", "% generated by paper/make_assets.py from data/ and results/; do not edit\n" +
          "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())))
    for k, v in sorted(macros.items()):
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
