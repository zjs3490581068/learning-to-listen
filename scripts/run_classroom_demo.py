"""Score the untouched official test set and build a portable A1 classroom demo."""

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from omegaconf import OmegaConf

from src.classroom_degradation import (
    METHODS, PEAK_LIMIT, PROTOCOL, RATE, SAMPLES, make_variants, pcm24_roundtrip,
    seeded_rng, select_pairs, shared_headroom, source_ranking, summarize_interventions, waveform_hash,
)
from src.data import AudioProcessor
from src.data_protocol import check_original_annotations, dataset_records, records_fingerprint
from src.inference import load_checkpoint
from src.model import MusicQualityModel
from src.config import load_config, add_paths_argument
from src.runtime import choose_gpu as select_gpu, bind_gpu
from train import set_seed


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_gpu(requested=None):
    return select_gpu(requested)[0]


def read_clip(dataset, index):
    item = dataset[index]["audio"]
    wave, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32", always_2d=True)
    return AudioProcessor().process(wave.mean(axis=1), sr, mode="center").numpy()


class Scorer:
    def __init__(self, cfg, checkpoint):
        self.model = MusicQualityModel(cfg)
        state = load_checkpoint(self.model, checkpoint, cfg, 0)
        self.provenance = state["data_protocol"]
        del state
        self.model.cuda().eval()

    @torch.inference_mode()
    def __call__(self, waves):
        results = []
        for start in range(0, len(waves), 16):
            array = np.stack(waves[start:start + 16])
            if array.shape[1:] != (SAMPLES,) or not np.isfinite(array).all():
                raise ValueError("Invalid scoring batch")
            tensor = torch.from_numpy(array).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = self.model(tensor, torch.zeros(len(tensor), dtype=torch.long, device="cuda"))
            values = {name: value.float().cpu().numpy() for name, value in scores.items()}
            if set(values) != {"MI", "TA"} or not all(np.isfinite(v).all() for v in values.values()):
                raise FloatingPointError("Non-finite or unexpected model output")
            results.extend({name: float(values[name][i]) for name in values} for i in range(len(tensor)))
        return results


def make_test_manifest(cfg):
    dataset = load_dataset(cfg.data.musiceval_id, split="test").cast_column("audio", Audio(decode=False))
    records = dataset_records(dataset)
    check_original_annotations(records, cfg.data.musiceval_metadata_dir, "test")
    training_manifest = json.loads(Path(cfg.data.split_manifest).read_text())
    if {r["audio_filename"] for r in records} & {r["audio_filename"] for r in training_manifest["records"]}:
        raise ValueError("Official test set overlaps training")
    if len(records) != 413 or len({r["model_id"] for r in records}) != 30:
        raise ValueError("Unexpected official test sample/system counts")
    import csv
    with (Path(cfg.data.musiceval_metadata_dir) / "prompt_info.txt").open(encoding="utf-8-sig") as stream:
        prompts = {r["id"]: r["text"] for r in csv.DictReader(stream, delimiter="\t") if r["id"]}
    with (Path(cfg.data.musiceval_metadata_dir) / "demo_prompt_info.txt").open(encoding="utf-8-sig") as stream:
        overrides = {r["id"]: r["text"] for r in csv.DictReader(stream, delimiter="\t") if r["id"]}
    for record in records:
        audio = dataset[record["clip_id"]]["audio"]
        info = sf.info(io.BytesIO(audio["bytes"]))
        prompt = Path(record["audio_filename"]).stem.split("_")[-1]
        text = overrides.get(record["audio_filename"], prompts.get(prompt))
        if not text:
            raise ValueError(f"Missing original prompt text: {record['audio_filename']}")
        record.update(prompt_id=prompt, prompt=text,
                      prompt_key=hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest(),
                      prompt_origin="per-audio override" if record["audio_filename"] in overrides else "common prompt table",
                      duration=info.duration,
                      original_audio_sha256=hashlib.sha256(audio["bytes"]).hexdigest(),
                      split="test", original_rate=info.samplerate,
                      crop_start_sec=max(0, (int(round(info.duration * RATE)) - SAMPLES) // 2 / RATE))
    threshold = float(np.quantile([r["mi_score"] for r in records], .75))
    pool = [r["clip_id"] for r in records if r["mi_score"] >= threshold and r["duration"] >= 10]
    if len(pool) != 127:
        raise ValueError(f"Expected fixed 127-source pool; got {len(pool)}")
    return dataset, {"protocol": PROTOCOL, "split": "test", "n_samples": 413, "n_systems": 30,
                     "records": records, "records_sha256": records_fingerprint(records),
                     "dataset_fingerprint": records_fingerprint(records), "candidate_indices": pool,
                     "human_MI_q75": threshold, "train_records_sha256": training_manifest["records_sha256"],
                     "train_overlap_count": 0}


def write_csv(path, rows):
    import csv
    if not rows:
        raise ValueError("Cannot export empty table")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def export_audio(root, relative, wave, score, **metadata):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.max(np.abs(wave)) > PEAK_LIMIT + 1e-6:
        raise ValueError("Audio exceeds the shared headroom ceiling")
    sf.write(path, wave, RATE, subtype="PCM_24")
    actual, rate = sf.read(path, dtype="float32")
    if rate != RATE or not np.array_equal(actual, wave):
        raise ValueError("Exported WAV differs from the scored PCM-24 samples")
    return {**metadata, **score, "path": relative, "waveform_sha256": waveform_hash(actual),
            "file_sha256": sha256_file(path), "samples": len(actual)}


def export_cases(root, dataset, baselines, interventions, scorer, selection):
    by_id = {r["clip_id"]: r for r in baselines}
    intervention_map = {(r["clip_id"], r["variant"]): r for r in interventions}
    payload = {"sources": [], "pairs": [], "limitations": [], "methods": METHODS}
    for clip_id in selection["sources"]:
        base = by_id[clip_id]
        waves, metadata = make_variants(read_clip(dataset, clip_id), base["audio_filename"])
        actual_scores = scorer(waves)
        variants = []
        for wave, meta, score in zip(waves, metadata, actual_scores):
            expected = intervention_map[(clip_id, meta["variant"])]
            if max(abs(score[h] - expected[h]) for h in ("MI", "TA")) > 1e-4:
                raise ValueError("Re-scoring selected PCM-24 audio disagrees with complete candidate results")
            variants.append(export_audio(root, f"audio/source_{clip_id}/{meta['variant']}.wav", wave,
                                         score, **meta))
        payload["sources"].append({"record": base, "variants": variants,
                                   "selection": "精选：人工高分、模型前四分位；优先单调方法数及重度降幅；系统和提示不重复"})
    for pair_number, pair in enumerate(selection["pairs"], 1):
        ids = [pair["high"], pair["low"]]
        waves, gain = shared_headroom([read_clip(dataset, i) for i in ids])
        waves = [pcm24_roundtrip(w) for w in waves]
        scores = scorer(waves)
        if scores[0]["MI"] <= scores[1]["MI"]:
            raise ValueError("Selected high-low ranking reversed on final exported waveforms")
        order = seeded_rng(str(pair_number), "pair_order").permutation(2)
        entries = []
        for letter, pos in zip(("A", "B"), order):
            entries.append(export_audio(root, f"audio/pair_{pair_number}/{letter}.wav", waves[pos], scores[pos],
                                        record=by_id[ids[pos]], letter=letter,
                                        category="较高分" if pos == 0 else "较低分", common_gain=gain))
        payload["pairs"].append({"name": f"对照 {pair_number}", "same_prompt": pair["same_prompt"],
                                 "entries": entries, "common_gain": gain})
    counter = selection.get("counterexample")
    if counter:
        clip_id, variant = counter["clip_id"], counter["variant"]
        base = by_id[clip_id]
        waves, metadata = make_variants(read_clip(dataset, clip_id), base["audio_filename"])
        chosen = [i for i, row in enumerate(metadata) if row["variant"] in ("clean", variant)]
        # Keep the same batch shape as the complete candidate run for bf16 reproducibility.
        scores = scorer(waves)
        entries = []
        for index in chosen:
            meta, score = metadata[index], scores[index]
            expected = intervention_map[(clip_id, meta["variant"])]
            if max(abs(score[h] - expected[h]) for h in ("MI", "TA")) > 1e-4:
                raise ValueError("Counterexample score changed")
            entries.append(export_audio(root, f"audio/limitation_{clip_id}/{meta['variant']}.wav", waves[index],
                                        score, **meta))
        payload["limitations"].append({"record": base, "variants": entries,
                                       "increase": -intervention_map[(clip_id, variant)]["drop_MI"]})
    return payload


def validate_selection(selection, automatic, baseline, rows):
    if len(selection["sources"]) != 3 or len(set(selection["sources"])) != 3 or len(selection["pairs"]) != 3:
        raise ValueError("A selection must contain three distinct sources and three pairs")
    eligible = {r["clip_id"] for r in automatic}
    if not set(selection["sources"]) <= eligible:
        raise ValueError("Manual sources must satisfy the fixed human/model eligibility gates")
    by_id = {r["clip_id"]: r for r in baseline}
    for key in ("model_id", "prompt_key"):
        if len({by_id[i][key] for i in selection["sources"]}) != 3:
            raise ValueError("Manual sources must retain system and actual-prompt diversity")
    low, high = np.quantile([r["MI"] for r in baseline], [.25, .75])
    used = set()
    for pair in selection["pairs"]:
        h, l = by_id[pair["high"]], by_id[pair["low"]]
        if (h["MI"] < high or l["MI"] > low or h["mi_score"] < 3.8 or l["mi_score"] > 2.6
                or min(h["duration"], l["duration"]) < 10 or used & {h["clip_id"], l["clip_id"]}):
            raise ValueError("Invalid or duplicate high-low pair")
        if pair["same_prompt"] != (h["prompt_key"] == l["prompt_key"]):
            raise ValueError("Incorrect same-prompt flag")
        used.update([h["clip_id"], l["clip_id"]])
    if selection.get("counterexample"):
        counter = selection["counterexample"]
        match = [r for r in rows if r["clip_id"] == counter["clip_id"] and r["variant"] == counter["variant"]]
        if len(match) != 1 or match[0]["drop_MI"] >= -1e-4:
            raise ValueError("Counterexample must be an actual score increase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A1_true_ttm.yaml")
    parser.add_argument("--output-dir", help="Default: storage.output_root/A1_classroom_degradation")
    parser.add_argument("--gpu", help="Physical GPU index or UUID, within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--reuse-scores", action="store_true", help="Validate and reuse complete fixed-protocol scores")
    parser.add_argument("--selection", help="Edited selection JSON; retains the same eligibility rules")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    if (cfg.model.tuning_mode != "frozen" or cfg.training.batch_size != 16
            or cfg.training.mixed_precision != "bf16" or cfg.seed != 42):
        raise ValueError("This experiment requires the approved A1 frozen/batch16/bf16/seed42 settings")
    checkpoint = Path(cfg.paths.output_dir) / cfg.experiment.name / "fold0/best_model.pt"
    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(cfg.storage.output_root) / "A1_classroom_degradation"
    if root.exists() and not args.reuse_scores:
        raise FileExistsError("Existing output: use --reuse-scores only for verified completed scoring")
    dataset, manifest = make_test_manifest(cfg)
    identity = {"protocol": PROTOCOL, "test_records_sha256": manifest["records_sha256"],
                "checkpoint_sha256": sha256_file(checkpoint), "checkpoint": f"{cfg.experiment.name}/fold0/best_model.pt",
                "training_manifest_sha256": sha256_file(cfg.data.split_manifest),
                "degradation_code_sha256": sha256_file(REPO / "src/classroom_degradation.py"),
                "methods": METHODS, "seed": 42, "batch_size": 16, "precision": "bf16",
                "sample_rate": RATE, "samples": SAMPLES, "peak_limit": PEAK_LIMIT}
    if args.reuse_scores and json.loads((root / "run_identity.json").read_text()) != identity:
        raise ValueError("Cached scores belong to different data, checkpoint, or degradation implementation")
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "run_identity.json", identity)
    save_json(root / "test_manifest.json", manifest)
    OmegaConf.save(cfg, root / "resolved_config.yaml")
    gpu = choose_gpu(args.gpu)
    print(json.dumps({"command": [sys.executable, *sys.argv], "cwd": str(REPO), "environment": sys.prefix,
                      "gpu": gpu, "allocator_limit_gib": 6, "output": str(root), "offline": True,
                      "training": False, "logs": "stdout/stderr redirected by launcher"}), flush=True)
    bind_gpu(gpu)
    torch.set_num_threads(4)
    set_seed(42)
    scorer = Scorer(cfg, checkpoint)
    started = time.time()
    status = {"status": "scoring", "gpu": gpu, "started": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "completed_sources": 0, "checkpoint_provenance": scorer.provenance}
    save_json(root / "run_status.json", status)
    try:
        if args.reuse_scores:
            baselines = json.loads((root / "baseline_predictions.json").read_text())
            interventions = json.loads((root / "degradation_predictions.json").read_text())
        else:
            baselines = []
            for start in range(0, 413, 16):
                records = manifest["records"][start:start + 16]
                scores = scorer([read_clip(dataset, r["clip_id"]) for r in records])
                baselines.extend({**record, **score} for record, score in zip(records, scores))
            save_json(root / "baseline_predictions.json", baselines)
            print("Official test baseline complete: 413 clips", flush=True)
            interventions = []
            for position, clip_id in enumerate(manifest["candidate_indices"], 1):
                base = baselines[clip_id]
                waves, metadata = make_variants(read_clip(dataset, clip_id), base["audio_filename"])
                scores = scorer(waves)
                for meta, score in zip(metadata, scores):
                    interventions.append({"clip_id": clip_id, "audio_filename": base["audio_filename"],
                                          "model_id": base["model_id"], "prompt_id": base["prompt_id"],
                                          **meta, **score, "raw_baseline_MI": base["MI"], "raw_baseline_TA": base["TA"],
                                          "drop_MI": scores[0]["MI"] - score["MI"],
                                          "drop_TA": scores[0]["TA"] - score["TA"]})
                status["completed_sources"] = position
                save_json(root / "run_status.json", status)
                if position % 10 == 0 or position == 127:
                    print(f"Degradation sources scored: {position}/127", flush=True)
            save_json(root / "degradation_predictions.json", interventions)
        if ([r["clip_id"] for r in baselines] != list(range(413)) or len(interventions) != 127 * 13
                or len({(r["clip_id"], r["variant"]) for r in interventions}) != 127 * 13
                or {r["clip_id"] for r in interventions} != set(manifest["candidate_indices"])):
            raise ValueError("Scoring results are duplicated or incomplete")
        summary = summarize_interventions(interventions)
        ids, ranking = source_ranking(baselines, interventions)
        save_json(root / "candidate_ranking.json", ranking)
        increases = sorted((r for r in interventions if r["drop_MI"] < -1e-4), key=lambda r: r["drop_MI"])
        selection = {"sources": ids, "pairs": select_pairs(baselines),
                     "counterexample": {"clip_id": increases[0]["clip_id"], "variant": increases[0]["variant"]} if increases else None,
                     "selection_note": "Automatic illustrative selection; not manually auditioned; complete pool retained"}
        if args.selection:
            selection = json.loads(Path(args.selection).read_text())
        validate_selection(selection, ranking, baselines, interventions)
        save_json(root / "selection.json", selection)
        payload = export_cases(root, dataset, baselines, interventions, scorer, selection)
        summary.update(n_baseline=413, n_sources=127, n_intervention_scores=len(interventions),
                       selection_note=selection["selection_note"], audition_status="Not manually auditioned",
                       model="Self-trained A1 fold0", split="official test", n_systems=30,
                       waveform_note="10s center crops; no loudness normalization; shared attenuation only",
                       human_note="Human MOS describes the original full recording, not the 10s or degraded versions",
                       paper_note="Classroom adaptation, not an exact reproduction of all paper degradation settings",
                       wall_time_sec=time.time() - started,
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        save_json(root / "summary.json", summary)
        save_json(root / "classroom_cases.json", payload)
        write_csv(root / "baseline_predictions.csv", baselines)
        write_csv(root / "degradation_predictions.csv", interventions)
        from src.classroom_report import build_report
        build_report(root, payload, summary, baselines, interventions)
        status.update(status="complete", completed_sources=127, wall_time_sec=time.time() - started,
                      finished=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        save_json(root / "run_status.json", status)
        print(json.dumps({"output": str(root), "selection": selection, "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        status.update(status="failed", error=repr(error))
        save_json(root / "run_status.json", status)
        raise


if __name__ == "__main__":
    main()
