"""Optional local-weight compatibility/zero-adapter check; no optimizer updates."""

import argparse
import gc
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import add_paths_argument, load_config
from src.inference import load_checkpoint
from src.model import MusicQualityModel
from src.personalization import add_personal_adapter, load_author, freeze_batchnorm, gradient_audit
from src.runtime import choose_gpu, bind_gpu


@torch.no_grad()
def forward(model, wave):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(wave)
    return {k: v.float().cpu().numpy() for k, v in output.items()}


def difference(left, right):
    if set(left) != set(right) or not all(np.isfinite(v).all() for v in (*left.values(), *right.values())):
        raise FloatingPointError("Nonfinite or incompatible diagnostic output")
    return max(float(np.max(np.abs(left[k] - right[k]))) for k in left)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A3a_personalized.yaml")
    parser.add_argument("--gpu")
    parser.add_argument("--a1-checkpoint", help="Optional verified A1 fold0 checkpoint")
    parser.add_argument("--reference-repo", help="Optional trusted original source tree for forward comparison")
    parser.add_argument("--backward", action="store_true", help="One diagnostic backward, no optimizer or weight updates")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    gpu, free = choose_gpu(args.gpu)
    print(json.dumps({"command": [sys.executable, *sys.argv], "gpu": gpu, "free_mib": free,
                      "allocator_limit_gib": 6, "optimizer_updates": 0, "offline": True}), flush=True)
    bind_gpu(gpu)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    reference = None
    if args.reference_repo:
        path = Path(args.reference_repo).resolve() / "src"
        spec = importlib.util.spec_from_file_location("muq_reference", path / "__init__.py", submodule_search_locations=[str(path)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        reference = importlib.import_module("muq_reference.model").MusicQualityModel
    wave = (.1 * torch.sin(2 * torch.pi * 440 * torch.arange(240000, dtype=torch.float32) / 24000)).unsqueeze(0).cuda()
    report = {}
    if args.a1_checkpoint:
        a1 = load_config("configs/A1_true_ttm.yaml", args.paths)
        model = MusicQualityModel(a1)
        state = load_checkpoint(model, args.a1_checkpoint, a1, 0)
        report["a1_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected = forward(model.cuda(), wave)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if reference:
            model = reference(a1)
            model.load_state_dict(state["model_state"], strict=True)
            report["a1_reference_max_abs_error"] = difference(expected, forward(model.cuda(), wave))
            del model
        del state
        gc.collect()
        torch.cuda.empty_cache()
    model = load_author(cfg).cuda()
    expected = forward(model, wave)
    scope = add_personal_adapter(model, cfg)
    report["personal_trainable_parameters"] = scope["trainable_parameters"]
    report["zero_adapter_max_abs_error"] = difference(expected, forward(model, wave))
    if args.backward:
        model.train()
        freeze_batchnorm(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(wave)["MI"].float().square().mean()
        loss.backward()
        report["gradient_audit"] = gradient_audit(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if reference:
        model = reference(cfg)
        state = torch.load(cfg.personalization.author_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        report["a3a_reference_max_abs_error"] = difference(expected, forward(model.cuda(), wave))
        del model
    for key, value in report.items():
        if key.endswith("max_abs_error") and value > 1e-6:
            raise AssertionError(f"Compatibility check failed: {key}={value}")
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3, optimizer_updates=0,
                  input="synthetic 440 Hz sine, 24 kHz mono 10 s", diagnostic_batch_size=1)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
