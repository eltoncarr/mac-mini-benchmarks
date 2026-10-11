#!/usr/bin/env python3
"""Make sure models/tokenizer has a working tokenizer.json for llama-benchy.

Some Qwen repos ship only vocab.json + merges.txt. Loaded through transformers
that can encode every text to zero tokens, which crashes llama-benchy. This
builds tokenizer.json from vocab.json + merges.txt with Qwen's standard
byte-level BPE pre-tokenizer, and can check it against llama-server's own
tokenizer (which reads the GGUF).

  tokenizer_fix.py ensure models/tokenizer
  tokenizer_fix.py check  models/tokenizer --server http://127.0.0.1:8080 --text book.txt
"""
import argparse
import json
import pathlib
import sys
import urllib.request

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

# Pre-tokenizer split used by Qwen2 through Qwen3.x tokenizer.json files.
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+")
SAMPLE = "Hello world, this is a test.\ndef area(r):\n    return 3.14159 * r ** 2  # 42 tokens?"


def n_tokens(path, text=SAMPLE):
    try:
        return len(Tokenizer.from_file(str(path)).encode(text, add_special_tokens=False).ids)
    except Exception:
        return 0


def build(d):
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    merges = []
    for line in (d / "merges.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#version"):
            a, b = line.split(" ", 1)
            merges.append((a, b))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=False))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.save(str(d / "tokenizer.json"))


def ensure(d):
    tj = d / "tokenizer.json"
    if tj.exists() and n_tokens(tj) > 5:
        print(f"tokenizer ok: {tj}")
        return 0
    if not ((d / "vocab.json").exists() and (d / "merges.txt").exists()):
        print(f"no usable tokenizer.json and no vocab.json/merges.txt in {d}", file=sys.stderr)
        return 1
    build(d)
    n = n_tokens(tj)
    if n <= 5:
        print(f"built {tj} but it encodes the sample to {n} tokens", file=sys.stderr)
        return 1
    print(f"built {tj} from vocab.json + merges.txt ({n} tokens for the sample)")
    return 0


def check(d, server, text_file, tolerance=0.02):
    """Compare local vs llama-server token counts on a slice of real text."""
    text = pathlib.Path(text_file).read_text(encoding="utf-8")[20000:36000] if text_file else SAMPLE * 50
    local = n_tokens(d / "tokenizer.json", text)
    req = urllib.request.Request(server.rstrip("/") + "/tokenize", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    remote = len(json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"])
    diff = abs(local - remote) / max(1, remote)
    print(f"tokenizer check: local {local} vs llama-server {remote} tokens ({100 * diff:.2f}% apart)")
    return 0 if local > 0 and diff <= tolerance else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["ensure", "check"])
    ap.add_argument("dir")
    ap.add_argument("--server")
    ap.add_argument("--text")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    sys.exit(ensure(d) if a.action == "ensure" else check(d, a.server, a.text))


if __name__ == "__main__":
    main()
