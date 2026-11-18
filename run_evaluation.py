"""
Evaluation benchmark for MuQ-Eval.

Loads a trained checkpoint and evaluates on MusicEval test folds,
reproducing Tables 1-2 from the paper.

Usage:
    # Evaluate a single fold
    python run_evaluation.py --checkpoint outputs/A1_frozen_mlp/fold0/best_model.pt \
                             --config configs/A1_frozen_mlp.yaml --fold 0

    # Evaluate all 5 folds and report mean +/- std
    python run_evaluation.py --checkpoint-dir outputs/A1_frozen_mlp \
                             --config configs/A1_frozen_mlp.yaml --all-folds

    # Evaluate with bootstrap CIs
    python run_evaluation.py --checkpoint-dir outputs/A1_frozen_mlp \
                             --config configs/A1_frozen_mlp.yaml --all-folds --bootstrap
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
from src.data import build_dataloaders
from src.model import MusicQualityModel
from src.inference import collect_predictions, evaluate_predictions, load_checkpoint
from src.config import load_config, add_paths_argument


def evaluate_fold(checkpoint_path, config, fold, device="cuda", bootstrap=False):
    """Evaluate a single fold and return metrics."""
    # Load model
    model = MusicQualityModel(config)
    state = load_checkpoint(model, checkpoint_path, config, fold)
    del state
    model = model.to(device).eval()

    # Build test dataloader
    _, val_loader, _ = build_dataloaders(config, fold_idx=fold)
    predictions = collect_predictions(model, val_loader, config, fold, device)
    results = evaluate_predictions(predictions, config, bootstrap)
    results["system_level"] = results["heads"]["MI"]["system"]
    for name in ("MI", "TA"):
        results[f"utterance_level_{name}"] = results["heads"][name]["utterance"]
    return results


def main():
    parser = argparse.ArgumentParser(description="MuQ-Eval evaluation benchmark")
    parser.add_argument("--checkpoint", type=str, help="Path to single checkpoint")
    parser.add_argument("--checkpoint-dir", type=str,
                        help="Directory with fold0/, fold1/, ... subdirectories")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate")
    parser.add_argument("--all-folds", action="store_true",
                        help="Evaluate all 5 folds")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute bootstrap CIs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: stdout)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", help="Physical GPU index or UUID within CUDA_VISIBLE_DEVICES")
    add_paths_argument(parser)
    args = parser.parse_args()

    config = load_config(args.config, args.paths)
    if args.device == "cuda":
        from src.runtime import choose_gpu, bind_gpu
        gpu, free = choose_gpu(args.gpu)
        print(f"Evaluation GPU {gpu}, free {free} MiB, allocator limit 6 GiB", flush=True)
        bind_gpu(gpu)
    if not args.checkpoint and not args.checkpoint_dir:
        args.checkpoint_dir = str(Path(config.paths.output_dir) / config.experiment.name)

    if args.all_folds:
        assert args.checkpoint_dir, "--checkpoint-dir required with --all-folds"
        fold_results = []
        for fold in range(config.data.cv_folds):
            ckpt = Path(args.checkpoint_dir) / f"fold{fold}" / "best_model.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing fold checkpoint: {ckpt}")
            print(f"Evaluating fold {fold}...")
            result = evaluate_fold(
                str(ckpt), config, fold, args.device, args.bootstrap
            )
            result["fold"] = fold
            fold_results.append(result)

        # Aggregate across folds
        aggregate = aggregate_fold_results(fold_results)
        output = {"folds": fold_results, "aggregate": aggregate}
    else:
        ckpt = args.checkpoint or str(
            Path(args.checkpoint_dir) / f"fold{args.fold}" / "best_model.pt"
        )
        output = evaluate_fold(ckpt, config, args.fold, args.device, args.bootstrap)
        output["fold"] = args.fold

    # Output
    output_str = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Results saved to {args.output}")
    else:
        print(output_str)


def aggregate_fold_results(fold_results):
    """Compute mean +/- std across folds."""
    agg = {}
    metric_keys = ["system_level", "utterance_level_MI", "utterance_level_TA"]

    for key in metric_keys:
        values_by_metric = {}
        for fr in fold_results:
            if key not in fr:
                continue
            for metric_name, value in fr[key].items():
                if isinstance(value, (int, float)):
                    values_by_metric.setdefault(metric_name, []).append(value)

        if values_by_metric:
            agg[key] = {}
            for metric_name, values in values_by_metric.items():
                arr = np.array(values)
                agg[key][metric_name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "values": [float(v) for v in arr],
                }

    return agg


if __name__ == "__main__":
    main()
