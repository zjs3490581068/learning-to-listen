"""Score a local music clip with the verified personal-preference checkpoint."""

import argparse
import io
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import soundfile as sf
import torch

from src.personalization import (
    add_personal_adapter, configure_gpu, identity_for, load_author,
    load_personal_checkpoint, preference_score, seed_model,
)
from src.personalization_data import RATE, SAMPLES, allowed_path, decode_source, wave_hash
from src.config import load_config, add_paths_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--start-sec", type=float, help="Default: centered ten seconds of the full recording")
    parser.add_argument("--prepared", action="store_true", help="Input is an already prepared 24kHz/mono/10s PCM-24 WAV")
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--config", default="configs/A3a_personalized.yaml")
    parser.add_argument("--gpu", help="Physical GPU index or UUID, within CUDA_VISIBLE_DEVICES")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    cfg.personalization.preferred_gpu = args.gpu
    root, path = allowed_path(cfg.paths.output_dir), allowed_path(args.audio)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    if args.prepared:
        if args.start_sec is not None:
            raise ValueError("--prepared cannot be combined with --start-sec")
        wave, rate = sf.read(path, dtype="float32")
        if rate != RATE or wave.shape != (SAMPLES,) or sf.info(path).subtype != "PCM_24":
            raise ValueError("Prepared input must be 24kHz mono, 10s, PCM-24")
        info, start = {"common_gain": 1., "common_gain_db": 0.}, 0
    else:
        wave, info = decode_source(path)
        start = (len(wave) - SAMPLES) // 2 if args.start_sec is None else round(args.start_sec * RATE)
        if start < 0 or start + SAMPLES > len(wave):
            raise ValueError("Requested ten-second interval is outside the recording; padding is not applied")
        wave = wave[start:start + SAMPLES] * info["common_gain"]
        buffer = io.BytesIO()
        sf.write(buffer, wave, RATE, format="WAV", subtype="PCM_24")
        buffer.seek(0)
        wave, _ = sf.read(buffer, dtype="float32")
    torch.set_num_threads(4)
    configure_gpu(cfg, "predict")
    seed_model(int(cfg.seed))
    model = load_author(cfg).cuda().eval()
    add_personal_adapter(model, cfg)
    checkpoint = root / ("personal_weights.pt" if args.lightweight else "best_model.pt")
    load_personal_checkpoint(model, checkpoint, identity_for(cfg, manifest))
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(torch.from_numpy(wave).unsqueeze(0).cuda())["MI"]
        score = float(preference_score(logits)[0])
    print(json.dumps({"audio": str(path), "start_sec": start / RATE, "end_sec": (start + SAMPLES) / RATE,
                      "preference_score": score, "meaning": "personal preference, not objective quality",
                      "common_gain": info["common_gain"], "common_gain_db": info["common_gain_db"],
                      "waveform_sha256": wave_hash(wave), "precision": "bf16", "batch_size": 1}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
