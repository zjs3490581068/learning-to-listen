"""Explicit GPU selection constrained by the caller's visible device set."""

import os
from pathlib import Path
import subprocess
import time


def source_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "source-export (upstream 990325ba8512a24b04ad2bb58149c02a494da975)"


def available_gpus(excluded=(), requested=None, minimum_free_gib=8):
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                    "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    permitted = None if visible is None else {v.strip() for v in visible.split(",") if v.strip()}
    wanted = None if requested is None else {v.strip() for v in str(requested).split(",") if v.strip()}
    for line in text.splitlines():
        parts = [v.strip() for v in line.split(",")]
        index, uuid, free = (int(parts[0]), parts[1], int(parts[2]))
        matches = lambda ids: ids is None or str(index) in ids or any(uuid.startswith(v) for v in ids if v.startswith("GPU-"))
        if matches(permitted) and matches(wanted) and index not in excluded and free >= minimum_free_gib * 1024:
            rows.append((index, free))
    return sorted(rows, key=lambda v: (-v[1], v[0]))


def choose_gpu(requested=None, minimum_free_gib=8):
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES disables GPUs; reference runs require CUDA")
    # A misspelled explicit selector must fail, not wait indefinitely.
    if not available_gpus(requested=requested, minimum_free_gib=0):
        raise RuntimeError("No GPU matches the requested/visible device set")
    while True:
        devices = available_gpus(requested=requested, minimum_free_gib=minimum_free_gib)
        if devices:
            return devices[0]
        print(f"Waiting for a visible GPU with >= {minimum_free_gib} GiB free", flush=True)
        time.sleep(30)


def bind_gpu(index, limit_gib=6):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Select the GPU before initializing CUDA")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_identifier(index)
    torch.cuda.set_per_process_memory_fraction(limit_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()


def gpu_identifier(index):
    identifier = subprocess.check_output(["nvidia-smi", "-i", str(index), "--query-gpu=uuid",
                                          "--format=csv,noheader"], text=True, timeout=15).strip()
    if not identifier.startswith("GPU-") or "\n" in identifier:
        raise RuntimeError("Unable to resolve physical GPU identity")
    return identifier
