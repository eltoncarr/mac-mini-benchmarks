#!/usr/bin/env python3
"""Write a mini-swe-agent SWE-bench config that points at the local server.

Starts from the swebench.yaml shipped with the installed mini-swe-agent so the
prompts and environment match the official SWE-bench bash-only setup; only the
model block and step limit are changed.
"""
import argparse
import pathlib

import minisweagent
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--api-base", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--step-limit", type=int, default=100)
ap.add_argument("--thinking", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()

pkg = pathlib.Path(minisweagent.__file__).parent
candidates = sorted(p for p in pkg.rglob("swebench.yaml") if "backticks" not in p.name)
if not candidates:
    raise SystemExit(f"swebench.yaml not found under {pkg}")
cfg = yaml.safe_load(candidates[0].read_text())

# Qwen3.8 recommended sampling: thinking (1.0/0.95/20) vs non-thinking (0.7/0.8/20).
sampling = ({"temperature": 1.0, "top_p": 0.95} if a.thinking else
            {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5})
model = cfg.setdefault("model", {})
model["model_name"] = f"openai/{a.model}"
model.setdefault("model_kwargs", {}).update({
    "api_base": a.api_base, "api_key": "local", "drop_params": True, **sampling,
    "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": bool(a.thinking)}},
})
model["cost_tracking"] = "ignore_errors"
cfg.setdefault("agent", {})["step_limit"] = a.step_limit
cfg["agent"]["cost_limit"] = 0  # local model: no cost limit

# SWE-bench images are x86_64 only; run them under Rosetta and allow for slow
# first pulls (images are 1-3 GB each) and emulated commands.
env = cfg.setdefault("environment", {})
env["run_args"] = ["--rm", "--platform", "linux/amd64"]
env["pull_timeout"] = 1800
env["timeout"] = max(int(env.get("timeout", 60)), 180)

pathlib.Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"base config: {candidates[0]}")
