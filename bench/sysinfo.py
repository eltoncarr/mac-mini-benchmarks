#!/usr/bin/env python3
"""Record machine, OS and software versions for a run (writes env.json). macOS or Linux + NVIDIA."""
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
from importlib import metadata


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def read(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return ""


def linux_machine():
    """Linux with an NVIDIA GPU (e.g. DGX Spark): same keys as on macOS where they apply."""
    osr = dict(line.split("=", 1) for line in read("/etc/os-release").splitlines() if "=" in line)
    rows = sh("nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader").splitlines()
    gpu = ([x.strip() for x in rows[0].split(",")] if rows else []) + ["", "", ""]
    mem_kb = next((int(x.split()[1]) for x in read("/proc/meminfo").splitlines() if x.startswith("MemTotal")), 0)
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", sh("nvidia-smi"))
    return {"os": osr.get("PRETTY_NAME", "").strip('"'), "kernel": platform.release(), "arch": platform.machine(),
            "model_name": read("/sys/class/dmi/id/product_name") or None, "chip": gpu[0],
            "gpu_driver": gpu[1], "gpu_power_limit": gpu[2],
            "cuda": cuda.group(1) if cuda else None, "memory_gb": round(mem_kb / 2**20),
            "cpu": (re.search(r"Model name:\s*(.+)", sh("lscpu")) or [None, None])[1],
            "cpu_cores": os.cpu_count(), "nvcc": (re.search(r"release ([\d.]+)", sh("nvcc", "--version")) or [None, None])[1]}


def main(out, llama_dir, model, extra):
    if platform.system() != "Darwin":
        machine = linux_machine()
        return write(out, machine, llama_dir, model, extra)
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
    write(out, machine, llama_dir, model, extra)


def write(out, machine, llama_dir, model, extra):
    pkgs = {}
    for p in ["mlx", "mlx-lm", "mini-swe-agent", "swebench", "evalplus", "llama-benchy", "litellm", "huggingface_hub"]:
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
