"""Regenerate ordered A1 validation predictions from verified best checkpoints."""

import argparse
import json
from pathlib import Path

from src.config import add_paths_argument, load_config
from src.data import build_dataloaders
from src.inference import collect_predictions, load_checkpoint
from src.model import MusicQualityModel
from src.runtime import bind_gpu, choose_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A1_true_ttm.yaml")
    parser.add_argument("--fold", type=int, choices=range(5), help="Default: all five folds")
    parser.add_argument("--gpu", help="Physical GPU index or UUID within CUDA_VISIBLE_DEVICES")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    gpu, free = choose_gpu(args.gpu)
    print(f"Inference GPU {gpu}, free {free} MiB, allocator limit 6 GiB", flush=True)
    bind_gpu(gpu)
    for fold in range(5) if args.fold is None else (args.fold,):
        root = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold{fold}"
        model = MusicQualityModel(cfg)
        checkpoint = load_checkpoint(model, root / "best_model.pt", cfg, fold)
        del checkpoint
        model.cuda().eval()
        _, loader, _ = build_dataloaders(cfg, fold)
        predictions = collect_predictions(model, loader, cfg, fold, "cuda")
        (root / "test_predictions.json").write_text(json.dumps(predictions, indent=2, allow_nan=False) + "\n")
        del model


if __name__ == "__main__":
    main()
