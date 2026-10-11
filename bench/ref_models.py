#!/usr/bin/env python3
"""Download the Llama-2 7B F16 / Q8_0 / Q4_0 GGUFs used by llama.cpp Discussion #4167.

Picks the files from the repo's actual file list (case-insensitive, BF16 and
repacked Q4_0_x_y variants excluded) instead of guessing name patterns.

  ref_models.py <dest-dir> [--repo ggml-org/Llama-2-7B-GGUF]
"""
import argparse
import pathlib
import sys

KINDS = ("F16", "Q8_0", "Q4_0")


def pick(files, kind):
    """Return the single-file GGUF for one quant kind, or None."""
    cands = []
    for f in files:
        name = pathlib.PurePosixPath(f).name.upper()
        if not name.endswith(".GGUF") or "MMPROJ" in name or "-OF-" in name:
            continue
        if kind == "F16" and ("FP16" in name or ("F16" in name and "BF16" not in name)):
            cands.append(f)
        elif kind == "Q8_0" and "Q8_0" in name:
            cands.append(f)
        elif kind == "Q4_0" and "Q4_0" in name and "Q4_0_" not in name:
            cands.append(f)
    return sorted(cands, key=len)[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--repo", default="ggml-org/Llama-2-7B-GGUF")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(a.repo)
    missing = []
    for kind in KINDS:
        f = pick(files, kind)
        if not f:
            missing.append(kind)
            continue
        path = hf_hub_download(a.repo, f, local_dir=a.dest)
        print(f"{kind}: {f} -> {path}")
    if missing:
        print(f"no {', '.join(missing)} file in {a.repo}; files: {', '.join(files)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
