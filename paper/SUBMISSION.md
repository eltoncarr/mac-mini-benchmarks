# arXiv submission: "Bytes per Token Set the Pace"

## Files

Upload `arxiv.tar.gz` (built by `make_arxiv.sh`; see "Build"). It contains `main.tex`, `main.bbl`
(arXiv does not run BibTeX), `references.bib`, `figures/*.pdf`, `generated/*.tex`, and ancillary
files in `anc/` (the dataset and `make_assets.py`), which arXiv lists as ancillary material.

## Metadata (paste into the arXiv form)

- **Title:** Bytes per Token Set the Pace: Benchmarking 3.5-bit and 8-bit Qwen3.8-27B on an Apple M5 Pro Mac mini
- **Primary category:** cs.PF (Performance). **Cross-lists:** cs.LG, cs.AR
- **License:** CC BY 4.0 (matches the data license)
- **Comments:** 11 pages, 4 figures, 5 tables. Data, per-run raw outputs and benchmark harness: https://github.com/eltoncarr/mac-mini-benchmarks
- **Abstract** (plain text, no LaTeX):

We present a controlled single-machine benchmark of two quantized builds of the dense Qwen3.8-27B language model running under llama.cpp on an Apple Mac mini with an M5 Pro system-on-chip (20-core GPU, 64 GB unified memory, 307 GB/s): a 3.5-bit-per-weight GSQ-RCO non-uniform quantization (IQ3_S, 12.12 GB) and a standard 8-bit quantization (Q8_0, 29.05 GB). Holding hardware, engine version and settings fixed, we measure decode and prefill throughput across context lengths of 0-64k tokens, the effect of multi-token-prediction (MTP) speculative decoding, time to first token with and without prompt caching, concurrent-request throughput, power, and coding accuracy on HumanEval+. Our thesis is that on this bandwidth-bound unified-memory device, decode speed is set by the bytes read per token, and the advantage of fewer bytes is bounded by compute headroom. IQ3_S decodes 1.97x faster than Q8_0 (19.96 vs. 10.12 tokens/s) with identical HumanEval+ pass@1 (89.6%, 95% confidence interval of the difference +/-6.6 points). The advantage narrows to 1.61x at 64k context, where a format-independent attention cost of about 0.35 ms per token per 1k tokens of context dominates; it narrows further to 1.10-1.41x with MTP, because Q8_0, being memory-bound, has more idle compute for speculative verification (MTP gain 1.32-1.75x vs. 1.04-1.27x). Prefill is compute-bound and format-neutral (400.1 vs. 414.8 tokens/s). All measurements, per-run raw outputs and the benchmark harness are released under open licenses.

## Before submitting

1. **Author block** in `main.tex`: replace "Author Name", "Affiliation" and the e-mail placeholder.
2. **Archive the data release** so the paper cites an immutable version:
   - merge the pending data fix (branch `fix-export-mtp-flag`), then tag the repo, e.g. `git tag v1.0.0 && git push origin v1.0.0`
   - optional but recommended: connect the repo to Zenodo and create a GitHub release, which mints a DOI
   - set `\datarelease` in `main.tex` to the tag (and add the DOI to the Data and Code Availability section and to `thisrepo` in `references.bib`)
3. **Acknowledgments:** the AI-assistance statement is a draft; edit it to match how you used AI tools. arXiv requires authors to take full responsibility for the content and does not accept AI tools as authors.
4. **Endorsement:** first-time submitters to cs.PF may need an endorsement from an existing author in that category.
5. Rebuild (below), check the PDF arXiv shows in the preview step, then submit.

## Build

```bash
python paper/make_assets.py      # regenerate figures, tables and numbers from data/ and results/
paper/make_arxiv.sh              # build main.pdf, then paper/arxiv.tar.gz, and test-compile it as arXiv does
```

Every number in the paper comes from `generated/numbers.tex` and the generated tables, which
`make_assets.py` computes from the released data; editing numbers by hand in `main.tex` breaks that link.
