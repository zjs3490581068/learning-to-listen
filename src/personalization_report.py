"""Offline preference listening page, generated independently of maintained docs."""

import json
from pathlib import Path

import numpy as np

from .classroom_report import make_waveform
from .personalization_data import allowed_path, save_json


def build_report(root):
    root = allowed_path(root)
    load = lambda name: json.loads((root / name).read_text())
    evaluation, training, manifest = (load(n) for n in ("evaluation_results.json", "training_results.json", "dataset_manifest.json"))
    history, predictions = load("training_history.json"), load("predictions.json")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for index, row in enumerate(evaluation["demonstrations"]):
        image = f"figures/demo_{index}.png"
        make_waveform(root / row["path"], root / image, "#1d715f" if index < 2 else "#247fa1")
        row["waveform_image"] = image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), layout="constrained")
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train CE", color="#247fa1")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="Validation CE", color="#b57820")
    axes[1].plot(epochs, [r["val_macro_mae"] for r in history], label="Validation macro MAE", color="#1d715f")
    axes[1].plot(epochs, [r["val_mae"] for r in history], label="Validation MAE", color="#b34855")
    for ax, ylabel in zip(axes, ["Gaussian ordinal CE", "Absolute preference-score error"]):
        ax.axvline(training["best_epoch"], color="#777777", ls="--", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(figures / "learning_curves.png", dpi=140)
    plt.close(fig)
    validation = [r for r in predictions if r["split"] == "validation"]
    labels = sorted({r["preference_target"] for r in validation})
    colors = ["#b34855", "#b57820", "#247fa1", "#1d715f", "#666666"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), layout="constrained", sharey=True)
    for ax, field, title in zip(axes, ("author_MI_score", "preference_score"), ("Author MI baseline", "Personal preference model")):
        for label, color in zip(labels, colors):
            rows = [r for r in validation if r["preference_target"] == label]
            ax.scatter(label + np.linspace(-.11, .11, len(rows)), [r[field] for r in rows], color=color, s=25, alpha=.8)
        ax.plot([1, 5], [1, 5], color="#777777", ls="--", lw=1)
        ax.set(title=title, xlabel="Assigned preference target", ylabel="Score", xticks=[1, 2, 3, 4, 5], ylim=(.8, 5.2))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    fig.savefig(figures / "validation_scores.png", dpi=140)
    plt.close(fig)
    counts = {split: sum(r["split"] == split for r in manifest["samples"]) for split in ("train", "validation")}
    page_data = {"demonstrations": evaluation["demonstrations"], "evaluation": evaluation,
                 "training": training, "source_count": len(manifest["personal_sources"]), "counts": counts,
                 "protocol": manifest["protocol"]}
    template = (Path(__file__).parents[1] / "templates/personalization.html").read_text()
    encoded = json.dumps(page_data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (root / "index.html").write_text(template.replace("__PERSONAL_DATA__", encoded))
    save_json(root / "presentation.json", page_data)
    text = ["# 个人偏好实验运行报告", "", f"协议：`{manifest['protocol']}`。", "",
            "这是个人偏好适配，不是客观音乐质量评价；展示片段未用于选模，同曲其他片段参与训练。", "",
            f"个人歌曲 {len(manifest['personal_sources'])} 首；训练 {counts['train']} 条，验证 {counts['validation']} 条。",
            f"最佳轮次 {training['best_epoch']}，实际训练 {training['epochs_run']} 轮。", "",
            "| 验证指标 | 作者 MI | 个人偏好 |", "|---|---:|---:|"]
    for key in ("macro_mae", "mae", "rmse", "srcc"):
        before, after = evaluation["author_baseline_metrics"]["validation"], evaluation["metrics"]["validation"]
        if key in before and key in after:
            text.append(f"| {key} | {before[key]:.6f} | {after[key]:.6f} |")
    text += ["", "作者 MI 与偏好目标语义不同；这里只比较其与个人标签的匹配程度。", "",
             "| 展示音频 | 区间（秒） | 偏好目标 | 作者 MI | 偏好分数 | 预设区间 |", "|---|---|---:|---:|---:|---|"]
    for row in evaluation["demonstrations"]:
        text.append(f"| {row['title']} | {row['start_sec']:.3f}-{row['end_sec']:.3f} | {row['preference_target']} | "
                    f"{row['author_MI_score']:.6f} | {row['preference_score']:.6f} | {row['display_target_interval']} |")
    text += ["", f"预设目标达标 {evaluation['display_targets_met']}/{len(evaluation['demonstrations'])}；不自动针对展示集调参。",
             "未进行主观听审。MusicEval 的 1 分是人为偏好标签，不是原 MOS。", "",
             "打开 `index.html` 可离线试听；`predictions.csv` 为完整预测，`clip_manifest.csv` 为切片清单。",
             "完整实验方法和网页说明见源码仓库 docs。此运行报告不会覆盖人工维护的 README。", ""]
    (root / "REPORT.md").write_text("\n".join(text))
    status = load("run_status.json")
    status.update(status="complete", presentation="index.html")
    save_json(root / "run_status.json", status)
