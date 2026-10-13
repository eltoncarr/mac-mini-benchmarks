#!/usr/bin/env bash
# Build paper/arxiv.tar.gz: LaTeX sources, the .bbl arXiv needs, figures, generated tables,
# and ancillary files (dataset + generator) under anc/. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null
stage="$(mktemp -d)"
mkdir -p "$stage/figures" "$stage/generated" "$stage/anc"
cp main.tex main.bbl references.bib "$stage/"
cp figures/*.pdf "$stage/figures/"
cp generated/*.tex "$stage/generated/"
cp make_assets.py ../data/qwen3.8-27b-mac-mini-m5-pro.json ../data/qwen3.8-27b-mac-mini-m5-pro.csv "$stage/anc/"
# check it compiles the way arXiv builds it: pdflatex with the bundled .bbl, no BibTeX
(cd "$stage" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && ! grep -qE "undefined|^!" main.log && rm -f main.aux main.log main.out main.pdf)
tar -czf arxiv.tar.gz -C "$stage" .
rm -rf "$stage"
echo "arxiv.tar.gz: $(du -h arxiv.tar.gz | cut -f1), $(tar -tzf arxiv.tar.gz | grep -vc '/$') files"
