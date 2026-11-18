"""Validate completed folds and summarize only the true-TTM A1 experiment."""

import csv
import json
from pathlib import Path

import numpy as np


def check_completed_fold(directory, fold, manifest):
    directory = Path(directory)
    training = json.loads((directory / "training_results.json").read_text())
    predictions = json.loads((directory / "test_predictions.json").read_text())
    evaluation = json.loads((directory / "evaluation_results.json").read_text())
    if not (directory / "best_model.pt").is_file():
        raise ValueError("Best checkpoint is missing")
    if predictions["clip_ids"] != manifest["folds"][fold]["val_indices"]:
        raise ValueError("Completed fold has different validation indices")
    if evaluation["n_systems"] != 31 or evaluation["n_samples"] != len(predictions["clip_ids"]):
        raise ValueError("Incomplete validation/system coverage")
    if training["gradient_check"] != {"frozen_encoder_has_gradients": False,
                                       "pooling_has_gradients": True, "heads_have_gradients": True}:
        raise ValueError("A1 gradient check failed")
    for epoch in training["train_history"]:
        if not all(np.isfinite(value) for value in epoch.values() if isinstance(value, (int, float))):
            raise ValueError("Non-finite training history")
    for name in ("MI", "TA"):
        for group in ("utterance", "system"):
            if not all(np.isfinite(v) for v in evaluation["heads"][name][group].values()):
                raise ValueError("Non-finite evaluation")
    return training, predictions, evaluation


def summarize_experiment(directory, manifest):
    directory = Path(directory)
    completed = [check_completed_fold(directory / f"fold{i}", i, manifest)
                 for i in range(manifest["n_folds"])]
    ids = [clip for _, pred, _ in completed for clip in pred["clip_ids"]]
    if sorted(ids) != list(range(manifest["n_samples"])):
        raise ValueError("OOF predictions are duplicated or incomplete")
    summary = {"protocol": manifest["protocol"], "records_sha256": manifest["records_sha256"],
               "n_folds": len(completed), "n_oof_samples": len(ids), "n_systems": 31,
               "selection_note": "Each CV validation fold also selected the best epoch; not an independent test set",
               "confidence_intervals": "95% BCa, B=1000, saved separately per fold; interval endpoints are not averaged",
               "paper_reference": {"arxiv": "2603.22677v2", "revision_date": "2026-10-28", "tables": [1, 2]},
               "protocol_note": "True TTM stratification differs from the original index-modulo fallback",
               "hyperparameter_note": "Repository lr=0.001, epochs=30, patience=7 retained; paper text gives 0.0003, 50, 10",
               "encoder_mode_note": "Frozen parameters; original train-mode BatchNorm running-buffer updates retained",
               "heads": {}, "folds": []}
    for name in ("MI", "TA"):
        summary["heads"][name] = {}
        for group, metrics in [("utterance", ("srcc", "pcc")),
                               ("system", ("srcc_sys", "pcc_sys", "ktau_sys"))]:
            summary["heads"][name][group] = {}
            for metric in metrics:
                values = [ev["heads"][name][group][metric] for _, _, ev in completed]
                summary["heads"][name][group][metric] = {
                    "mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    rows = []
    for fold, (training, predictions, _) in enumerate(completed):
        summary["folds"].append({"fold": fold, "best_epoch": training["best_epoch"],
                                  "epochs_run": len(training["train_history"]),
                                  "wall_time_sec": training["wall_time_sec"],
                                  "resources": training["resources"]})
        for pos, clip in enumerate(predictions["clip_ids"]):
            record = manifest["records"][clip]
            if (predictions["model_ids"][pos] != record["model_id"]
                    or predictions["audio_filenames"][pos] != record["audio_filename"]):
                raise ValueError("OOF identity mismatch")
            rows.append({"clip_id": clip, "fold": fold, "model_id": record["model_id"],
                         "audio_filename": record["audio_filename"],
                         **{f"{key}_{name}": predictions[f"{key}_{name}"][pos]
                            for key in ("predictions", "targets") for name in ("MI", "TA")}})
    rows.sort(key=lambda row: row["clip_id"])
    with (directory / "oof_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["resources"] = {
        "training_process_wall_time_sec_sum": sum(t["wall_time_sec"] for t, _, _ in completed),
        "max_peak_allocated_gib": max(t["resources"]["peak_allocated_gib"] for t, _, _ in completed),
        "max_peak_reserved_gib": max(t["resources"]["peak_reserved_gib"] for t, _, _ in completed),
        "checkpoint_bytes": sum((directory / f"fold{i}/best_model.pt").stat().st_size
                                for i in range(manifest["n_folds"])),
    }
    summary["paper_comparison"] = {}
    for label, head, group, metric, paper, paper_std in [
            ("MI_system_SRCC", "MI", "system", "srcc_sys", 0.957, 0.014),
            ("MI_system_PCC", "MI", "system", "pcc_sys", 0.960, 0.013),
            ("MI_system_Kendall_tau", "MI", "system", "ktau_sys", 0.839, 0.019),
            ("MI_utterance_SRCC", "MI", "utterance", "srcc", 0.838, 0.016),
            ("MI_utterance_PCC", "MI", "utterance", "pcc", 0.828, 0.017),
            ("TA_utterance_SRCC", "TA", "utterance", "srcc", 0.591, 0.034),
            ("TA_utterance_PCC", "TA", "utterance", "pcc", 0.594, 0.041)]:
        value = summary["heads"][head][group][metric]
        summary["paper_comparison"][label] = {"paper": paper, "paper_std": paper_std,
                                                "measured": value["mean"], "measured_std": value["std"],
                                                "difference": value["mean"] - paper}
    (directory / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = ["# A1 True-TTM Reproduction", "", "Official train split: 1,923 clips; true system-stratified five-fold CV.",
            "Validation folds also selected checkpoints; these are not independent-test estimates.", "",
            "| Head | Level | Metric | Five-fold mean | SD |", "|---|---|---|---:|---:|"]
    for name, groups in summary["heads"].items():
        for group, metrics in groups.items():
            for metric, value in metrics.items():
                text.append(f"| {name} | {group} | {metric} | {value['mean']:.6f} | {value['std']:.6f} |")
    text += ["", "Per-fold 95% BCa confidence intervals are in fold*/evaluation_results.json.",
             summary["protocol_note"] + ".", summary["hyperparameter_note"] + ".",
             summary["encoder_mode_note"] + ".", "", "## Paper Comparison", "",
             "Reference: arXiv:2603.22677v2, revised 2026-10-28, Tables 1-2. V2 does not report TA system scores.", "",
             "| Metric | Paper mean +/- SD | Reproduction mean +/- SD | Difference |",
             "|---|---:|---:|---:|"]
    for label, comparison in summary["paper_comparison"].items():
        text.append(f"| {label} | {comparison['paper']:.3f} +/- {comparison['paper_std']:.3f} "
                    f"| {comparison['measured']:.6f} +/- {comparison['measured_std']:.6f} "
                    f"| {comparison['difference']:+.6f} |")
    text += ["", "## Resources", "",
             f"Peak PyTorch allocated: {summary['resources']['max_peak_allocated_gib']:.3f} GiB; "
             f"reserved: {summary['resources']['max_peak_reserved_gib']:.3f} GiB.",
             f"Five full best checkpoints: {summary['resources']['checkpoint_bytes'] / 1024**3:.3f} GiB.", "",
             "| Fold | Epochs run | Best epoch (zero-based) | Training wall time (s) |",
             "|---|---:|---:|---:|"]
    for fold in summary["folds"]:
        text.append(f"| {fold['fold']} | {fold['epochs_run']} | {fold['best_epoch']} | {fold['wall_time_sec']:.2f} |")
    (directory / "reproduction_summary.md").write_text("\n".join(text) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for fold, (training, _, _) in enumerate(completed):
        history = training["train_history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["val/MI_srcc"] for row in history], label=f"Fold {fold}")
        axes[1].plot(epochs, [row["val/loss"] for row in history], label=f"Fold {fold}")
    for ax, title, ylabel in [(axes[0], "MI validation correlation", "SRCC"),
                              (axes[1], "Validation loss", "MI + TA MSE")]:
        ax.set(xlabel="Epoch (zero-based)", ylabel=ylabel, title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    figure.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(figure)
    return summary
