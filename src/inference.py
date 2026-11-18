"""Shared checkpoint loading, aligned predictions, and per-fold evaluation."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data_protocol import PROTOCOL, records_fingerprint
from .evaluation import full_evaluation


def load_checkpoint(model, path, cfg, fold_idx=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = cfg.data.get("split_manifest")
    if expected:
        manifest = json.loads(Path(expected).read_text())
        provenance = checkpoint.get("data_protocol", {})
        if (provenance.get("protocol") != PROTOCOL
                or provenance.get("records_sha256") != manifest["records_sha256"]
                or provenance.get("manifest_sha256") != records_fingerprint(manifest)
                or (fold_idx is not None and provenance.get("fold") != fold_idx)):
            raise ValueError("Checkpoint does not belong to the configured split manifest")
    state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    return checkpoint


@torch.inference_mode()
def collect_predictions(model, loader, cfg, fold_idx, device):
    model.eval()
    device = torch.device(device)
    result = {"fold": fold_idx, "experiment": cfg.experiment.name,
              "protocol": PROTOCOL, "clip_ids": [], "model_ids": [],
              "audio_filenames": [], "predictions_MI": [], "predictions_TA": [],
              "targets_MI": [], "targets_TA": []}
    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16
    amp = device.type == "cuda" and cfg.training.mixed_precision != "no"
    for batch in tqdm(loader, desc=f"Predict fold {fold_idx}"):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            predictions = model(batch["waveform"].to(device), batch["dataset_id"].to(device))
        result["clip_ids"].extend(batch["clip_id"].tolist())
        result["model_ids"].extend(batch["model_id"])
        result["audio_filenames"].extend(batch["audio_filename"])
        for name, target in [("MI", "mi_score"), ("TA", "ta_score")]:
            pred = predictions[name]
            if pred.ndim > 1:
                centers = torch.linspace(1, 5, pred.shape[-1], device=pred.device)
                pred = (torch.softmax(pred, -1) * centers).sum(-1)
            values = pred.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Non-finite {name} predictions")
            result[f"predictions_{name}"].extend(values.tolist())
            result[f"targets_{name}"].extend(batch[target].tolist())
    result["n_samples"] = len(result["clip_ids"])
    if result["clip_ids"] != loader.dataset.indices:
        raise ValueError("Prediction order does not match validation indices")
    if len(set(result["clip_ids"])) != result["n_samples"]:
        raise ValueError("Duplicate prediction clip IDs")
    manifest_path = cfg.data.get("split_manifest")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        if result["clip_ids"] != manifest["folds"][fold_idx]["val_indices"]:
            raise ValueError("Predictions do not match saved fold")
        for position, clip_id in enumerate(result["clip_ids"]):
            record = manifest["records"][clip_id]
            if (result["model_ids"][position] != record["model_id"]
                    or result["audio_filenames"][position] != record["audio_filename"]):
                raise ValueError("Prediction identity mismatch")
            if not np.allclose([result["targets_MI"][position], result["targets_TA"][position]],
                               [record["mi_score"], record["ta_score"]], rtol=0, atol=1e-6):
                raise ValueError("Prediction target mismatch")
        result["records_sha256"] = manifest["records_sha256"]
        result["manifest_sha256"] = records_fingerprint(manifest)
    result["predictions"] = result["predictions_MI"]
    result["targets"] = result["targets_MI"]
    return result


def evaluate_predictions(predictions, cfg, bootstrap=True):
    results = {"fold": predictions["fold"], "protocol": PROTOCOL,
               "n_samples": predictions["n_samples"],
               "n_systems": len(set(predictions["model_ids"])),
               "selection_note": "CV validation fold used for both early stopping and evaluation; not independent test",
               "heads": {}}
    for name in ("MI", "TA"):
        results["heads"][name] = full_evaluation(
            np.asarray(predictions[f"predictions_{name}"]),
            np.asarray(predictions[f"targets_{name}"]),
            np.asarray(predictions["model_ids"]),
            n_bootstrap=cfg.evaluation.bootstrap_n if bootstrap else 0,
            ci_level=cfg.evaluation.bootstrap_ci)
    return results
