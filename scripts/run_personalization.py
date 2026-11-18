"""Prepare, train, evaluate, or publish the fixed personal-preference experiment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf
import torch

from src.personalization import evaluate_personal, train_personal
from src.personalization_data import allowed_path, prepare_data
from src.config import load_config, add_paths_argument
from src.config import scientific_config
from src.runtime import source_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A3a_personalized.yaml")
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "report", "all"), default="all")
    parser.add_argument("--gpu", help="Physical GPU index or UUID, within CUDA_VISIBLE_DEVICES")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    cfg.personalization.preferred_gpu = args.gpu
    if (cfg.seed != 42 or cfg.training.batch_size != 16 or cfg.training.mixed_precision != "bf16"
            or cfg.training.epochs != 30 or cfg.training.warmup_steps != 24 or cfg.model.tuning_mode != "lora"
            or cfg.loss.type != "ordinal_ce" or cfg.loss.ordinal_sigma != .5):
        raise ValueError("Configuration differs from the approved fixed personalization protocol")
    torch.set_num_threads(4)
    root = allowed_path(cfg.paths.output_dir)
    print(json.dumps({"command": [sys.executable, *sys.argv], "cwd": str(REPO), "environment": sys.prefix,
                      "stage": args.stage, "output": str(root), "offline": True,
                      "source_revision": source_revision()}), flush=True)
    manifest = prepare_data(cfg)
    config_path = root / "resolved_config.yaml"
    if config_path.exists():
        if scientific_config(OmegaConf.load(config_path)) != scientific_config(cfg):
            raise ValueError("Existing experiment configuration differs")
    else:
        OmegaConf.save(cfg, config_path)
    if args.stage in ("train", "all"):
        train_personal(cfg, manifest)
    if args.stage in ("evaluate", "all"):
        evaluate_personal(cfg, manifest)
    if args.stage in ("report", "all"):
        from src.personalization_report import build_report
        build_report(root)


if __name__ == "__main__":
    main()
