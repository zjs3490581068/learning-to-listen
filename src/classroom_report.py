"""Self-contained offline listening UI and transparent full-pool reports."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .classroom_degradation import METHODS


def make_waveform(path, target, color):
    from PIL import Image, ImageDraw
    wave, _ = sf.read(path, dtype="float32")
    peaks = np.max(np.abs(wave.reshape(600, -1)), axis=1)
    image = Image.new("RGB", (600, 96), "#f5f7f8")
    draw = ImageDraw.Draw(image)
    for x, peak in enumerate(peaks):
        half = max(1, int(peak * 45))
        draw.line((x, 48 - half, x, 48 + half), fill=color)
    image.save(target)


def build_report(root, payload, summary, baselines, interventions):
    root = Path(root)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1d715f", "#247fa1", "#ad791b", "#c84957"]
    for group in payload["sources"] + payload["limitations"]:
        for entry in group["variants"]:
            relative = "figures/" + entry["path"].replace("/", "_").replace(".wav", ".png")
            make_waveform(root / entry["path"], root / relative, colors[entry["severity"]])
            entry["waveform_image"] = relative
    for pair in payload["pairs"]:
        for entry in pair["entries"]:
            relative = "figures/" + entry["path"].replace("/", "_").replace(".wav", ".png")
            make_waveform(root / entry["path"], root / relative, colors[1])
            entry["waveform_image"] = relative
    for source in payload["sources"]:
        source["charts"] = {}
        for method in METHODS:
            entries = [e for e in source["variants"] if e["method"] in ("clean", method)]
            fig, ax = plt.subplots(figsize=(7.6, 2.8), layout="constrained")
            values = [e["MI"] for e in entries]
            ax.plot(range(4), values, color=colors[0], marker="o", lw=2)
            for i, value in enumerate(values):
                ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 9), textcoords="offset points", ha="center")
            ax.set(xticks=range(4), xticklabels=["Reference", "Mild", "Moderate", "Severe"],
                   ylabel="MI score", ylim=(min(values) - .25, max(values) + .3))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            relative = f"figures/source_{source['record']['clip_id']}_{method}_curve.png"
            fig.savefig(root / relative, dpi=130)
            plt.close(fig)
            source["charts"][method] = relative
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    for i, (method, results) in enumerate(summary["methods"].items()):
        axes[0].plot([1, 2, 3], [results["levels"][str(s)]["mean_drop_MI"] for s in (1, 2, 3)],
                     marker="o", label=method, color=colors[i])
        axes[1].plot([1, 2, 3], [results["levels"][str(s)]["decrease_rate"] for s in (1, 2, 3)],
                     marker="o", label=method, color=colors[i])
    for ax, ylabel in [(axes[0], "Mean MI drop (reference - degraded)"), (axes[1], "Fraction with MI decrease")]:
        ax.set(xticks=[1, 2, 3], xticklabels=["Mild", "Moderate", "Severe"], ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    axes[0].axhline(0, color="#777777", lw=.8)
    axes[1].set_ylim(0, 1.05)
    fig.savefig(figures / "complete_pool.png", dpi=150)
    plt.close(fig)
    data = {"cases": payload, "summary": summary}
    template = (Path(__file__).parents[1] / "templates/classroom.html").read_text()
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (root / "index.html").write_text(template.replace("__CLASSROOM_DATA__", encoded))
    (root / "classroom_cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    text = ["# A1 课堂受控退化实验", "", "## 数据与模型", "",
            f"自训 A1 fold0；官方 test {summary['n_baseline']} 条、{summary['n_systems']} 个系统，与训练集无文件名重叠。",
            f"退化候选为人工 MI 前四分位且长度至少 10 秒的全部 {summary['n_sources']} 条，不按退化分数删样。",
            "24 kHz 单声道中心 10 秒；不做响度归一化，只做整组一致的防溢出衰减。",
            "所有退化评分经过 PCM-24 往返；精选 WAV 重读并重新推理，与完整结果核对。",
            "人工 MOS 只标注了原始完整录音，不是裁剪片段或退化版本的新人工评分。",
            "这是课堂实验扩展，不是论文所有退化设置的逐项严格复现。", "", "## 全部候选统计", "",
            "分数差定义：共享增益参考版 MI 减退化 MI；正值表示降分。平分阈值为 0.0001。",
            "单调率要求参考→轻→中→重每一步都严格降分。比例区间为 Wilson 95%。", "",
            "| 方法 | 强度 | 样本数 | 平均 MI 降幅 | 降分比例 | Wilson 95% | 严格单调率 |",
            "|---|---|---:|---:|---:|---|---:|"]
    for _, results in summary["methods"].items():
        for severity, level in results["levels"].items():
            ci = level["wilson_95"]
            text.append(f"| {results['label']} | {severity} | {level['n']} | {level['mean_drop_MI']:.4f} "
                        f"| {level['decrease_rate']:.1%} | [{ci[0]:.1%}, {ci[1]:.1%}] "
                        f"| {results['strict_monotonic_rate']:.1%} |")
    text += ["", "## 精选退化源", "", "按固定条件选出的成功展示案例不能代表完整统计；未进行人工听审。", "",
             "| 源文件 | 人工原始 MI | 原始模型 MI | 共享增益参考 MI | 增益 dB |",
             "|---|---:|---:|---:|---:|"]
    for source in payload["sources"]:
        record, clean = source["record"], source["variants"][0]
        text.append(f"| {record['audio_filename']} | {record['mi_score']:.1f} | {record['MI']:.3f} "
                    f"| {clean['MI']:.3f} | {clean['common_gain_db']:.2f} |")
    text += ["", "## 高低分原始音频", "",
             "| 对照 | 同提示 | A 文件 / MI / 原始人工 MI | B 文件 / MI / 原始人工 MI |", "|---|---|---|---|"]
    for pair in payload["pairs"]:
        entries = [f"{e['record']['audio_filename']} / {e['MI']:.3f} / {e['record']['mi_score']:.1f}"
                   for e in pair["entries"]]
        text.append(f"| {pair['name']} | {pair['same_prompt']} | {entries[0]} | {entries[1]} |")
    text += ["", "## 模型局限", ""]
    for example in payload["limitations"]:
        text.append(f"- {example['record']['audio_filename']}：{example['variants'][1]['variant']} "
                    f"比参考版升分 {example['increase']:.3f}。保留该结果，不作参数调整。")
    if not payload["limitations"]:
        text.append("未出现超过平分阈值的反向升分；非单调结果仍保留在完整数据表。")
    text += ["", "## 文件与重现", "",
             "- `index.html`：直接打开；整个目录可复制到其他电脑，无运行时联网。",
             "- `audio/`：最终评分的精选 PCM-24 WAV；`figures/`：波形与曲线。",
             "- `baseline_predictions.csv/json`：全部 413 条未经展示增益处理的原始评分。",
             "- `degradation_predictions.csv/json`：127×13 条完整参考版和退化评分，包括升分与平分。",
             "- `selection.json` 与 `candidate_ranking.json`：选择清单和排序依据。",
             "- `run_identity.json`、`test_manifest.json`：模型、处理代码、测试样本校验值。",
             "", "替换案例时编辑选择清单的副本，然后运行：", "", "```bash",
             "# Run from the source tree using the installed environment",
             "python scripts/run_classroom_demo.py --reuse-scores --selection /absolute/path/to/selection.json",
             "```", "", "该命令复用完整候选评分，只重新导出并核验最终案例，不重新训练。",
             f"评分及精选导出阶段耗时 {summary['wall_time_sec']:.1f} 秒（不含模型加载与报告生成）；CUDA 峰值分配 "
             f"{summary['peak_allocated_gib']:.3f} GiB、保留 {summary['peak_reserved_gib']:.3f} GiB。", "",
             "## 验收与交付", "",
             "`verification_results.json` 记录数据、波形、GPU 重读评分和浏览器验收结果；",
             "`verification/` 保存桌面、平板及手机截图和单元测试日志。人工听审尚未进行。",
             "源码仓库 docs 保留完整实验方法；此运行报告不会覆盖人工维护的 README。", ""]
    (root / "REPORT.md").write_text("\n".join(text))
