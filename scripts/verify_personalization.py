"""Audit fixed preference data, training provenance, exported scores, and offline UI."""

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets import Audio, load_dataset
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import soundfile as sf
import torch

from src.data_protocol import check_original_annotations, dataset_records, records_fingerprint
from src.personalization import identity_for, metrics
from src.personalization_data import (
    RATE, SAMPLES, allowed_path, choose_intervals, decode_source, file_hash, rng, save_json,
    validate_manifest, wave_hash,
)
from src.config import load_config, add_paths_argument
from src.config import scientific_config


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_csv(path, rows):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream))
    require(len(actual) == len(rows), f"CSV row count differs: {path.name}")
    for left, right in zip(actual, rows):
        for key, value in right.items():
            expected = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            require(left[key] == ("" if value is None else expected), f"CSV field differs: {path.name}/{key}")


class LocalAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href") and value and not value.startswith("#"):
                require(not urlparse(value).scheme, f"External resource: {value}")
                self.paths.add(value)


def compare_numeric(left, right, tolerance=1e-10):
    if isinstance(left, dict):
        require(set(left) == set(right), "Metric fields differ")
        for key in left:
            compare_numeric(left[key], right[key], tolerance)
    else:
        require(np.isfinite([left, right]).all() and abs(left - right) <= tolerance, "Metric differs")


def check_wave(root, row, wave, gain):
    audio = np.asarray(wave[row["start_sample_24k"]:row["end_sample_24k"]] * gain, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, RATE, format="WAV", subtype="PCM_24")
    buffer.seek(0)
    actual, rate = sf.read(buffer, dtype="float32")
    require(rate == RATE and wave_hash(actual) == row["waveform_sha256"], "Regenerated final WAV differs")
    require(file_hash(root / row["path"]) == row["file_sha256"], "WAV file hash differs")


def audit_sources(root, cfg, manifest):
    profile = manifest["music_profile"]
    songs = {s["title"]: s for s in profile["songs"]}
    rows = manifest["samples"] + manifest["demonstrations"]
    for source in manifest["personal_sources"]:
        wave, info = decode_source(allowed_path(Path(cfg.personalization.music_dir) / source["source_path"]))
        require(all(info[k] == source[k] for k in info), "Personal source preprocessing changed")
        selected, selection = choose_intervals(wave, songs[source["title"]].get("protected_interval"), profile["clips_per_song"])
        require(selection == source["selection"], "Energy selection protocol changed")
        group = sorted((r for r in manifest["samples"] if r["source_id"] == source["source_id"]),
                       key=lambda r: r["start_sec"])
        require(selected == [{k: row[k] for k in selected[0]} for row in group], "Selected intervals changed")
        val_slots = set(rng(f"split:{source['audio_filename']}", cfg.seed).permutation(profile["clips_per_song"])[-profile["validation_per_song"]:].tolist())
        for slot, row in enumerate(group):
            require(row["split"] == ("validation" if slot in val_slots else "train"), "Song split changed")
        for row in (r for r in rows if r["source_id"] == source["source_id"]):
            require(row["source_sha256"] == source["source_sha256"] and row["common_gain"] == info["common_gain"],
                    "Same-source provenance/gain changed")
            check_wave(root, row, wave, info["common_gain"])
    for split, count in (("train", profile["musiceval_count"]), ("test", 1)):
        dataset = load_dataset(cfg.data.musiceval_id, split=split).cast_column("audio", Audio(decode=False))
        records = dataset_records(dataset)
        check_original_annotations(records, cfg.personalization.metadata_dir, split)
        provenance = manifest["musiceval_provenance"][split]
        require(records_fingerprint(records) == provenance["records_sha256"], "Official data fingerprint changed")
        eligible = []
        for record in sorted(records, key=lambda r: r["audio_filename"]):
            audio = dataset[record["clip_id"]]["audio"]
            buffer = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else allowed_path(audio["path"])
            if sf.info(buffer).duration >= 10:
                eligible.append(record)
        chosen = rng(f"musiceval:{split}", cfg.seed).choice(len(eligible), count, replace=False)
        actual = [r for r in rows if r["source_kind"] == "musiceval" and r["official_split"] == split]
        for slot, (row, position) in enumerate(zip(actual, chosen)):
            record = eligible[int(position)]
            require(row["audio_filename"] == record["audio_filename"], "Random MusicEval selection changed")
            require(row["model_id"] == record["model_id"] and row["official_clip_id"] == record["clip_id"],
                    "MusicEval identity changed")
            require(row["preference_target"] == 1 and row["original_human_MI"] == record["mi_score"]
                    and row["original_human_TA"] == record["ta_score"], "Original/assigned labels changed")
            require(row["split"] == ("demo" if split == "test" else "train" if slot < count - profile["musiceval_validation"] else "validation"),
                    "MusicEval fixed split changed")
            audio = dataset[record["clip_id"]]["audio"]
            blob = audio["bytes"] if audio.get("bytes") is not None else allowed_path(audio["path"]).read_bytes()
            require(hashlib.sha256(blob).hexdigest() == row["source_sha256"], "MusicEval source audio changed")
            wave, info = decode_source(io.BytesIO(blob))
            require(all(info[k] == row[k] for k in info), "MusicEval preprocessing changed")
            require(row["start_sample_24k"] == (len(wave) - SAMPLES) // 2, "MusicEval crop is not centered")
            check_wave(root, row, wave, info["common_gain"])
    return {"status": "passed", "personal_sources": len(songs), "official_train": profile["musiceval_count"], "official_test": 1,
            "regenerated_wavs": len(rows), "random_selection_reproduced": True}


def audit(root, cfg):
    load = lambda name: json.loads((root / name).read_text())
    manifest, training, evaluation = load("dataset_manifest.json"), load("training_results.json"), load("evaluation_results.json")
    validate_manifest(manifest, root, music_dir=cfg.personalization.music_dir)
    identity = identity_for(cfg, manifest)
    require(all(load(name)["identity"] == identity for name in ("training_results.json", "evaluation_results.json")),
            "Configuration/data/checkpoint identity changed")
    require(load("run_identity.json") == identity, "Run identity differs")
    require(scientific_config(OmegaConf.load(root / "resolved_config.yaml")) == scientific_config(cfg),
            "Saved configuration differs")
    rows, history = load("predictions.json"), load("training_history.json")
    index = {r["sample_id"]: r for r in rows}
    originals = manifest["samples"] + manifest["demonstrations"]
    require(len(rows) == len(index) == len(originals), "Prediction duplicates or omissions")
    for original in originals:
        row = index[original["sample_id"]]
        require(all(row[k] == value for k, value in original.items()), "Prediction metadata changed")
        require(np.isfinite([row["author_MI_score"], row["preference_score"], row["absolute_error"]]).all(), "Nonfinite score")
        require(1 - 1e-6 <= row["preference_score"] <= 5 + 1e-6, "Invalid score expectation")
        require(abs(row["absolute_error"] - abs(row["preference_score"] - row["preference_target"])) < 1e-12,
                "Incorrect per-sample error")
    for split in ("train", "validation", "demo"):
        group = [r for r in rows if r["split"] == split]
        expected_order = [r["sample_id"] for r in originals if r["split"] == split]
        require([r["sample_id"] for r in group] == expected_order, "Prediction order changed")
        for field, key in (("preference_score", "metrics"), ("author_MI_score", "author_baseline_metrics")):
            compare_numeric(metrics([r["preference_target"] for r in group], [r[field] for r in group]),
                            evaluation[key][split])
    require([r["epoch"] for r in history] == list(range(1, len(history) + 1)), "Nonconsecutive training epochs")
    require(all(np.isfinite(list(r.values())).all() for r in history), "Nonfinite training history")
    selected = min(history, key=lambda r: (r["val_macro_mae"], r["val_mae"], r["epoch"]))
    require(selected["epoch"] == training["best_epoch"] == evaluation["selected_epoch"], "Wrong best-model selection")
    steps = (sum(r["split"] == "train" for r in manifest["samples"]) + cfg.training.batch_size - 1) // cfg.training.batch_size
    require(training["epochs_run"] == len(history) <= 30 and training["optimizer_steps"] == len(history) * steps,
            "Training schedule differs")
    require(training["warmup_steps"] == 24 and training["scheduled_maximum_steps"] == 30 * steps, "Warmup schedule differs")
    require(len(history) == 30 or len(history) - training["best_epoch"] == 7, "Unexpected early stopping")
    require(not training["demonstrations_used_for_selection"] and not evaluation["demonstrations_used_for_selection"],
            "Demonstration leakage into model selection")
    require(training["frozen_state_unchanged"] and training["batchnorm_unchanged"], "Frozen model state changed")
    require(training["initialization_max_abs_error"] <= 1e-6 and training["reload_max_abs_error"] <= 1e-4,
            "Initialization/reloading audit failed")
    require(training["scope"]["trainable_parameters"] == 1445638 and len(training["scope"]["modules"]) == 16,
            "Wrong LoRA parameter/module count")
    require(training["gradient_check"] == {"frozen_parameters_have_gradients": False,
            "lora_gradient_tensors": 32, "pooling_gradient_tensors": 4, "MI_head_gradient_tensors": 6},
            "Gradient audit failed")
    require(training["peak_allocated_gib"] <= 6 + .01, "Allocator limit exceeded")
    for key in ("fresh_reload_max_abs_error", "wav_reread_max_abs_error", "lightweight_max_abs_error"):
        require(evaluation[key] <= 1e-4, f"Model inference audit failed: {key}")
    windows = [r["display_target_interval"] for r in manifest["demonstrations"]]
    for row, original, window in zip(evaluation["demonstrations"], manifest["demonstrations"], windows):
        require(row == index[original["sample_id"]], "Demo scores differ from full table")
        require(row["display_target_interval"] == window, "Predeclared target changed")
        require(row["display_target_met"] == (window[0] <= row["preference_score"] <= window[1]), "Incorrect target claim")
    require(sum(r["display_target_met"] for r in evaluation["demonstrations"]) == evaluation["display_targets_met"],
            "Incorrect demonstration success count")
    compare_numeric(training["best_validation"], evaluation["metrics"]["validation"], 1e-4)
    check_csv(root / "predictions.csv", rows)
    check_csv(root / "demonstration_scores.csv", evaluation["demonstrations"])
    check_csv(root / "clip_manifest.csv", originals)
    check_csv(root / "training_history.csv", history)
    for filename, kind in (("best_model.pt", "personal_full_v1"), ("personal_weights.pt", "personal_light_v1")):
        checkpoint = torch.load(root / filename, map_location="cpu", weights_only=True)
        require(checkpoint["identity"] == identity and checkpoint["format"] == kind
                and checkpoint["epoch"] == training["best_epoch"], "Checkpoint provenance differs")
        if kind == "personal_light_v1":
            require(sum(w.numel() for w in checkpoint["weights"].values()) == 1445638, "Light export weight count differs")
        del checkpoint
    parser = LocalAssets()
    parser.feed((root / "index.html").read_text())
    for path in parser.paths:
        require(allowed_path(root / path).is_relative_to(root) and (root / path).is_file(), f"Missing resource: {path}")
    presentation = load("presentation.json")
    require(presentation["evaluation"]["metrics"] == evaluation["metrics"], "Web metrics differ")
    for row, expected in zip(presentation["demonstrations"], evaluation["demonstrations"]):
        require(all(row[k] == value for k, value in expected.items()), "Web demo differs")
        require((root / row["waveform_image"]).is_file(), "Missing waveform")
    for path in (root / "figures").glob("*.png"):
        pixels = np.asarray(Image.open(path).convert("RGB"))
        require(pixels.std() > 5 and np.mean(np.min(pixels, axis=2) < 220) > .005, f"Blank figure: {path}")
    require(load("run_status.json")["status"] == "complete", "Experiment has not completed")
    sources = audit_sources(root, cfg, manifest)
    return {"status": "passed", "data_and_predictions": {"samples": len(manifest['samples']),
            "train": sum(r['split'] == 'train' for r in manifest['samples']),
            "validation": sum(r['split'] == 'validation' for r in manifest['samples']),
            "demonstrations": len(manifest['demonstrations']), "scored_wavs": len(rows)}, "sources": sources,
            "training_and_reload_audits": "passed", "predeclared_targets_met": evaluation["display_targets_met"],
            "predeclared_targets_total": len(manifest['demonstrations']), "manual_audition": "not_performed"}


def browser_audit(root):
    from playwright.sync_api import sync_playwright
    expected = json.loads((root / "presentation.json").read_text())
    demo_count = len(expected["demonstrations"])
    target = root / "verification"
    target.mkdir(exist_ok=True)
    errors, external, screenshots = [], [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage",
                                                         "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("request", lambda request: external.append(request.url)
                if urlparse(request.url).scheme not in ("file", "data", "blob") else None)
        page.goto((root / "index.html").as_uri(), wait_until="load")
        require(page.locator("audio").count() == demo_count, "Missing demonstration players")
        page.wait_for_function("Array.from(document.querySelectorAll('audio')).every(a=>a.readyState>=2&&Math.abs(a.duration-10)<.01)")
        require(page.locator(".num").first.evaluate("e=>getComputedStyle(e).visibility") == "hidden", "Scores not blind by default")
        page.locator("#reveal").check()
        require(page.locator(".num").first.is_visible(), "Scores cannot be revealed")
        payload = page.locator("#personal-data").text_content()
        require(json.loads(payload) == json.loads((root / "presentation.json").read_text()), "Embedded web data changed")
        for i in range(demo_count):
            player = page.locator("audio").nth(i)
            player.evaluate("async a=>{await a.play();a.currentTime=2;}")
            page.wait_for_timeout(100)
            require(not player.evaluate("a=>a.paused"), f"Audio {i} cannot play")
            require(page.locator("audio").evaluate_all("aa=>aa.filter(a=>!a.paused).length") == 1, "Multiple tracks playing")
        page.locator("#tab-analysis").click()
        require(page.locator("audio").evaluate_all("aa=>aa.every(a=>a.paused)"), "Tab switch did not stop audio")
        require(page.locator("#class-metrics tr").count() == len(expected["evaluation"]["metrics"]["validation"]["per_class"]), "Missing class statistics")
        for name, width, height in (("desktop", 1440, 1000), ("tablet", 768, 1024), ("mobile", 390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            for tab in ("analysis", "listening"):
                page.locator(f"#tab-{tab}").click()
                require(page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"), f"Layout overflows: {name}/{tab}")
                require(page.locator("img").evaluate_all("aa=>aa.every(a=>a.complete&&a.naturalWidth>0)"), "Missing image")
                relative = f"verification/{name}_{tab}.png"
                page.screenshot(path=str(root / relative), full_page=True)
                screenshots.append(relative)
        page.locator("#reveal").uncheck()
        require(page.locator(".num").first.evaluate("e=>getComputedStyle(e).visibility") == "hidden", "Score hiding failed")
        page.screenshot(path=str(target / "mobile_blind.png"), full_page=True)
        screenshots.append("verification/mobile_blind.png")
        require(not external and not errors, f"Browser network/errors: {external}/{errors}")
        browser.close()
    for relative in screenshots:
        require(np.asarray(Image.open(root / relative).convert("RGB")).std() > 10, "Blank browser screenshot")
    return {"status": "passed", "offline_file_url": True, "external_requests": external, "errors": errors,
            "ten_second_tracks_played": demo_count, "single_track_playback": True, "score_toggle": True,
            "viewports": [1440, 768, 390], "screenshots": screenshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A3a_personalized.yaml")
    parser.add_argument("--browser", action="store_true")
    add_paths_argument(parser)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = load_config(args.config, args.paths)
    root = allowed_path(cfg.paths.output_dir)
    result = audit(root, cfg)
    result["browser"] = browser_audit(root) if args.browser else {"status": "not_run"}
    save_json(root / "verification_results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
