"""Read-only full decoding and MuQ input validation for a personal music folder."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.data import AudioProcessor
from src.config import load_config, add_paths_argument

EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".wma", ".mgg", ".mgg1", ".mgg2"}


def checked_path(path):
    return Path(path).expanduser().resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_file(path, processor=None):
    path = checked_path(path)
    processor = processor or AudioProcessor()
    with path.open("rb") as stream:
        head = stream.read(16)
    row = {"filename": path.name, "path": str(path), "bytes": path.stat().st_size,
           "sha256": file_hash(path), "header_hex": head.hex(), "status": "rejected", "warnings": []}
    if path.suffix.lower() == ".flac" and not head.startswith(b"fLaC"):
        row["warnings"].append("Not a plain FLAC header; suffix is not proof of decoded audio")
    try:
        info = sf.info(path)
        row.update(format=info.format, subtype=info.subtype, sample_rate=info.samplerate,
                   channels=info.channels, declared_frames=info.frames, duration_sec=info.duration)
        # Decode the entire recording, not only the header or a short preview.
        array, rate = sf.read(path, dtype="float32", always_2d=True)
        if (len(array) != info.frames or len(array) == 0 or rate != info.samplerate
                or array.shape[1] != info.channels or not np.isfinite(array).all()):
            raise ValueError("Decoded audio is empty, truncated, non-finite, or inconsistent with its header")
        row["decoded_frames"] = len(array)
        row["decoded_peak"] = float(np.max(np.abs(array)))
        row["fraction_above_full_scale"] = float(np.mean(np.abs(array) > 1))
        mono = array.mean(axis=1, dtype=np.float32)
        del array
        energy = sum(float(np.sum(part.astype(np.float64) ** 2))
                     for part in np.array_split(mono, max(1, math.ceil(len(mono) / 1000000))))
        row["mono_rms"] = math.sqrt(energy / len(mono))
        if row["mono_rms"] < 1e-5:
            raise ValueError("The decoded/downmixed recording is near silent")
        if row["fraction_above_full_scale"] > 0:
            row["warnings"].append("Decoded samples exceed +/-1; this may be codec overshoot, not a decoding failure")
        row["full_decode_ok"] = True
        processed = {}
        old_random_state = random.getstate()
        try:
            for mode in ("center", "random"):
                random.seed(int(row["sha256"][:16], 16) ^ 42)
                tensor = processor.process(mono, rate, mode=mode)
                if tensor.shape != (processor.clip_samples,) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid {mode} model input tensor")
                rms = float(torch.sqrt(torch.mean(tensor.double().square())))
                if rms < 1e-5:
                    raise ValueError(f"The selected {mode} model input is near silent")
                processed[mode] = tensor
                row[f"{mode}_rms"] = rms
                row[f"{mode}_peak"] = float(tensor.abs().max())
                row[f"{mode}_sha256"] = hashlib.sha256(tensor.numpy().astype("<f4").tobytes()).hexdigest()
        finally:
            random.setstate(old_random_state)
        row.update(status="usable", input_sample_rate=processor.target_sr,
                   input_samples=processor.clip_samples, input_dtype="float32", input_channels=1,
                   padding_samples=max(0, processor.clip_samples - math.ceil(len(mono) * processor.target_sr / rate)),
                   center_crop_start_sec=max(0, (math.ceil(len(mono) * processor.target_sr / rate)
                                                - processor.clip_samples) // 2 / processor.target_sr))
        return row, processed
    except (RuntimeError, ValueError, OSError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if path.suffix.lower() == ".flac" and not head.startswith(b"fLaC"):
            row["diagnosis"] = "Not directly decodable as ordinary audio; possibly still encrypted or incompletely converted. Not decrypted or repaired."
        return row, None


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A3a_personalized.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    source = checked_path(args.input_dir or cfg.personalization.music_dir)
    output = checked_path(args.output_dir or Path(cfg.storage.output_root) / "music_input_audit")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit; select a new --output-dir")
    paths = sorted(checked_path(p) for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError("No candidate audio files found")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    rows, inputs = [], {"center": [], "random": []}
    for number, path in enumerate(paths, 1):
        row, tensors = audit_file(path)
        rows.append(row)
        if tensors:
            for mode, tensor in tensors.items():
                inputs[mode].append(tensor)
        print(f"{number}/{len(paths)} {row['status']}: {path.name}", flush=True)
    batches = {}
    for mode, tensors in inputs.items():
        if tensors:
            batch = torch.stack(tensors)
            if not torch.isfinite(batch).all() or batch.shape[1] != 240000:
                raise ValueError("Collated model input failed validation")
            batches[mode] = {"shape": list(batch.shape), "dtype": str(batch.dtype), "finite": True}
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[row["sha256"]].append(row["filename"])
    summary = {"created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "source": str(source),
               "total_files": len(rows), "counts": dict(Counter(r["status"] for r in rows)),
               "usable_duration_sec": sum(r["duration_sec"] for r in rows if r["status"] == "usable"),
               "usable_formats": dict(Counter(r["format"] for r in rows if r["status"] == "usable")),
               "exact_duplicate_groups": [names for names in duplicates.values() if len(names) > 1],
               "input_batches": batches, "preprocessing": "Full soundfile decode, channel mean, original AudioProcessor resample/crop/pad; no loudness normalization",
               "environment": {"python": sys.executable, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                               "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                               "audio_backends": torchaudio.list_audio_backends(), "flac_supported": "FLAC" in sf.available_formats()},
               "wall_time_sec": time.perf_counter() - started, "source_files_modified": False,
               "gpu_used": False, "model_forward_performed": False, "training_performed": False}
    write_json(output / "audit.json", {"summary": summary, "files": rows})
    write_json(output / "usable_files.json", [r for r in rows if r["status"] == "usable"])
    write_json(output / "rejected_files.json", [r for r in rows if r["status"] == "rejected"])
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    text = ["# 本地音乐输入体检", "", f"检查时间：{summary['created']}。源目录：`{source}`。", "",
            f"共 {len(rows)} 个文件，可用 {summary['counts'].get('usable', 0)} 个，拒绝 {summary['counts'].get('rejected', 0)} 个。",
            "完整解码成功的文件经过单声道转换、24 kHz 重采样、中心和固定种子随机 10 秒裁剪，并检查有限 float32 张量和批次。",
            "仅执行 CPU 解码和预处理；未运行模型前向、未训练、未修改或导出音乐。", "",
            "## 可用文件", "", "| 文件 | 格式 | 原采样率 | 声道 | 全曲时长（秒） | 中心 10 秒 RMS |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] == "usable":
            text.append(f"| {row['filename']} | {row['format']}/{row['subtype']} | {row['sample_rate']} | {row['channels']} | {row['duration_sec']:.3f} | {row['center_rms']:.6f} |")
    text += ["", "## 无法直接使用", "", "以下文件实际解码失败；文件名后缀不是音频格式有效性的证据。非标准 FLAC 文件头及 mgg 字样提示可能仍加密或未完成转换，但本次不进行解密，也不把该推断当作已确认的损坏原因。", ""]
    for row in rows:
        if row["status"] == "rejected":
            text.append(f"- `{row['filename']}`：{row['error']}")
    text += ["", "## 后续使用", "", "- 可用文件清单为 `usable_files.json`；逐项信息、SHA256 和解码错误见 `audit.json` / `audit.csv`。",
             "- 无效文件需要从有权使用的来源重新导出为普通 WAV、FLAC、MP3 或 Ogg；只修改扩展名不能验证内容。",
             "- A3a 与当前 A1 共享波形输入预处理；输入可用并不意味着已经完成个性化训练。",
             "- 原 MusicEval 训练数据类要求官方标注及系统文件名；个人音乐后续需要单独的数据加载和偏好标注，不能直接冒充 MusicEval 样本。",
             "- 未进行主观听审，也未把未标注音乐自动当作高分或低分训练样本。", ""]
    (output / "README.md").write_text("\n".join(text))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
