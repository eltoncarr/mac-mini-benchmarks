#!/usr/bin/env python3
"""Record machine, OS and software versions for a run (writes env.json)."""
import json
import platform
import subprocess
import sys
from importlib import metadata


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def main(out, llama_dir, model, extra):
    machine = {"macos": platform.mac_ver()[0], "arch": platform.machine(),
               "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
               "memory_gb": round(int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30),
               "cpu_cores_perf": sh("sysctl", "-n", "hw.perflevel0.physicalcpu"),
               "cpu_cores_eff": sh("sysctl", "-n", "hw.perflevel1.physicalcpu"),
               "gpu_wired_limit_mb": sh("sysctl", "-n", "iogpu.wired_limit_mb"),
               "power_mode": sh("pmset", "-g")}
    try:
        hw = json.loads(sh("system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"))
        h = hw["SPHardwareDataType"][0]
        machine["model_name"] = h.get("machine_name")
        machine["model_id"] = h.get("machine_model")
        g = hw["SPDisplaysDataType"][0]
        machine["gpu_cores"] = g.get("sppci_cores")
    except Exception:
        pass
    pkgs = {}
    for p in ["mlx", "mlx-lm", "mini-swe-agent", "swebench", "evalplus", "litellm", "huggingface_hub"]:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    env = {"model": model, "machine": machine, "python": sys.version.split()[0], "packages": pkgs,
           "llama_cpp_commit": sh("git", "-C", llama_dir, "rev-parse", "--short", "HEAD"),
           "llama_cpp_describe": sh("git", "-C", llama_dir, "describe", "--tags", "--always"),
           **json.loads(extra or "{}")}
    with open(out, "w") as f:
        json.dump(env, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
