"""Single-fold A1 worker. Use scripts/run_a1_true_ttm.py for verified scheduling."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from src.config import load_config, add_paths_argument
from src.data import build_dataloaders
from src.inference import collect_predictions, evaluate_predictions, load_checkpoint
from src.losses import CombinedLoss
from src.model import MusicQualityModel
from src.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_fold(cfg, fold_idx):
    set_seed(cfg.seed + fold_idx)
    fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    fold_cfg.paths.output_dir = str(Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold{fold_idx}")
    fold_cfg.data.fold_idx = fold_idx
    fold_dir = Path(fold_cfg.paths.output_dir)
    if (fold_dir / "training_results.json").exists() or (fold_dir / "best_model.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing fold: {fold_dir}")
    train_loader, val_loader, extra_loader = build_dataloaders(fold_cfg, fold_idx)
    model = MusicQualityModel(fold_cfg)
    trainer = Trainer(fold_cfg, model, train_loader, val_loader, CombinedLoss(fold_cfg), extra_loader)
    results = trainer.train()
    checkpoint = load_checkpoint(model, fold_dir / "best_model.pt", fold_cfg, fold_idx)
    del checkpoint
    model.zero_grad(set_to_none=True)
    predictions = collect_predictions(model, val_loader, fold_cfg, fold_idx, trainer.device)
    evaluation = evaluate_predictions(predictions, fold_cfg)
    for name, data in (("test_predictions.json", predictions), ("evaluation_results.json", evaluation)):
        (fold_dir / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return results


def aggregate_folds(cfg, n_folds):
    base = Path(cfg.paths.output_dir) / cfg.experiment.name
    rows = [json.loads((base / f"fold{i}/training_results.json").read_text()) for i in range(n_folds)]
    result = {"n_folds": len(rows), "best_epochs": [r["best_epoch"] for r in rows]}
    for key in rows[0]["best_metrics"]:
        values = [row["best_metrics"][key] for row in rows]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    (base / "aggregated_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A1_true_ttm.yaml")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--max-gpu-memory-gib", type=float, default=6)
    parser.add_argument("--gpu", help="Physical GPU index or UUID within CUDA_VISIBLE_DEVICES")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    if args.aggregate_only:
        aggregate_folds(cfg, cfg.data.cv_folds)
        return
    from src.runtime import choose_gpu, bind_gpu
    gpu, free = choose_gpu(args.gpu)
    print(f"A1 fold {args.fold}: GPU {gpu}, free {free} MiB, allocator limit {args.max_gpu_memory_gib} GiB", flush=True)
    bind_gpu(gpu, args.max_gpu_memory_gib)
    run_fold(cfg, args.fold)


if __name__ == "__main__":
    main()
