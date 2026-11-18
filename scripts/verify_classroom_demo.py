"""Audit the complete classroom experiment, with optional GPU and offline-browser checks."""

import argparse
import csv
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image
import soundfile as sf

from scripts.run_classroom_demo import (
    Scorer, choose_gpu, load_config, make_test_manifest, read_clip, save_json, set_seed,
    sha256_file, validate_selection,
)
from src.classroom_degradation import (
    METHODS, PEAK_LIMIT, RATE, SAMPLES, TOLERANCE, make_variants,
    source_ranking, summarize_interventions, waveform_hash,
)
from src.data_protocol import records_fingerprint
from src.config import add_paths_argument


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class LocalAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href") and value and not value.startswith("#"):
                require(not urlparse(value).scheme, f"External HTML resource: {value}")
                self.paths.add(value)


def check_csv(path, rows):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream))
    require(len(actual) == len(rows), f"CSV row count differs: {path.name}")
    for left, right in zip(actual, rows):
        for key, value in right.items():
            expected = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            if value is None:
                expected = ""
            require(left[key] == expected, f"CSV differs from JSON: {path.name}/{key}")


def audit(root, cfg):
    def load(name):
        return json.loads((root / name).read_text())

    identity, manifest = load("run_identity.json"), load("test_manifest.json")
    baseline, rows = load("baseline_predictions.json"), load("degradation_predictions.json")
    cases, selection, summary = load("classroom_cases.json"), load("selection.json"), load("summary.json")
    dataset, current = make_test_manifest(cfg)
    require(current == manifest, "Current test data/prompt metadata differs from saved manifest")
    require(records_fingerprint(manifest["records"]) == identity["test_records_sha256"], "Test fingerprint mismatch")
    require(sha256_file(identity["checkpoint"]) == identity["checkpoint_sha256"], "Checkpoint changed")
    require(sha256_file(cfg.data.split_manifest) == identity["training_manifest_sha256"], "Training protocol changed")
    require(sha256_file(REPO / "src/classroom_degradation.py") == identity["degradation_code_sha256"], "Degradation code changed")
    require(load("run_status.json")["status"] == "complete", "Scoring did not complete")
    require([r["clip_id"] for r in baseline] == list(range(413)), "Baseline order/coverage mismatch")
    require(len({r["audio_filename"] for r in baseline}) == 413, "Repeated test filenames")
    for row, record in zip(baseline, manifest["records"]):
        require(all(row[k] == v for k, v in record.items()), "Baseline metadata mismatch")
        require(np.isfinite([row["MI"], row["TA"]]).all(), "Non-finite baseline")
    index = {(r["clip_id"], r["variant"]): r for r in rows}
    expected_variants = {"clean"} | {f"{method}_{level}" for method in METHODS for level in (1, 2, 3)}
    require(len(rows) == len(index) == 127 * 13, "Candidate scores duplicated or missing")
    require({r["clip_id"] for r in rows} == set(manifest["candidate_indices"]), "Candidate pool changed")
    regenerated = {}
    retained_sources = set(selection["sources"])
    if selection.get("counterexample"):
        retained_sources.add(selection["counterexample"]["clip_id"])
    for clip_id in manifest["candidate_indices"]:
        original = baseline[clip_id]
        wave = read_clip(dataset, clip_id)
        waves, metadata = make_variants(wave, original["audio_filename"])
        if clip_id in retained_sources:
            regenerated[clip_id] = (waves, metadata)
        actual = {r["variant"] for r in rows if r["clip_id"] == clip_id}
        require(actual == expected_variants, f"Missing variant: {clip_id}")
        clean = index[(clip_id, "clean")]
        for meta in metadata:
            row = index[(clip_id, meta["variant"])]
            require(all(row[k] == v for k, v in meta.items()), "Deterministic waveform metadata changed")
            require(row["audio_filename"] == original["audio_filename"] and row["model_id"] == original["model_id"],
                    "Candidate identity changed")
            require(np.isfinite([row[h] for h in ("MI", "TA", "drop_MI", "drop_TA")]).all(), "Non-finite candidate")
            for head in ("MI", "TA"):
                require(abs(row[f"drop_{head}"] - (clean[head] - row[head])) < 1e-12, "Incorrect paired drop")
                require(row[f"raw_baseline_{head}"] == original[head], "Incorrect raw baseline")
    computed = summarize_interventions(rows)
    require(all(summary[k] == v for k, v in computed.items()), "Full-pool summary differs from complete predictions")
    _, ranking = source_ranking(baseline, rows)
    require(load("candidate_ranking.json") == ranking, "Source ranking mismatch")
    validate_selection(selection, ranking, baseline, rows)
    require([s["record"]["clip_id"] for s in cases["sources"]] == selection["sources"], "Source selection mismatch")
    require(len({s["record"]["model_id"] for s in cases["sources"]}) == 3, "Repeated source system")
    require(len({s["record"]["prompt_key"] for s in cases["sources"]}) == 3, "Repeated source prompt")
    entries, checked_images = [], set()
    for group in cases["sources"] + cases["limitations"]:
        original = group["record"]
        require(original == baseline[original["clip_id"]], "Exported source metadata differs")
        for entry in group["variants"]:
            expected = index[(original["clip_id"], entry["variant"])]
            for head in ("MI", "TA", "waveform_sha256", "common_gain"):
                require(entry[head] == expected[head], "Exported candidate/score mismatch")
            entries.append(entry)
        checked_images.update(group.get("charts", {}).values())
    require(len(cases["sources"]) == 3 and all(len(s["variants"]) == 13 for s in cases["sources"]),
            "Expected three complete 13-waveform source groups")
    used = set()
    for pair, requested in zip(cases["pairs"], selection["pairs"]):
        require(len(pair["entries"]) == 2 and {e["letter"] for e in pair["entries"]} == {"A", "B"}, "Invalid A/B")
        ids = {e["record"]["clip_id"] for e in pair["entries"]}
        require(ids == {requested["high"], requested["low"]} and not ids & used, "Pair identity/uniqueness mismatch")
        require(pair["same_prompt"] == requested["same_prompt"], "Same-prompt flag mismatch")
        scores = {e["record"]["clip_id"]: e["MI"] for e in pair["entries"]}
        require(scores[requested["high"]] > scores[requested["low"]], "Final high-low ranking reversed")
        for entry in pair["entries"]:
            require(entry["record"] == baseline[entry["record"]["clip_id"]], "Pair record differs")
            require(entry["common_gain"] == pair["common_gain"], "Pair gain differs")
        used.update(ids)
        entries.extend(pair["entries"])
    require(len(cases["pairs"]) == 3, "Expected three high-low pairs")
    if selection["counterexample"]:
        require(len(cases["limitations"]) == 1, "Counterexample missing")
        example = cases["limitations"][0]
        chosen = selection["counterexample"]
        require(example["record"]["clip_id"] == chosen["clip_id"], "Wrong counterexample source")
        require(example["variants"][1]["variant"] == chosen["variant"], "Wrong counterexample variant")
        require(example["increase"] == -index[(chosen["clip_id"], chosen["variant"])]["drop_MI"], "Wrong increase")
    assets = LocalAssets()
    assets.feed((root / "index.html").read_text())
    maximum_peak = 0.
    for entry in entries:
        path = root / entry["path"]
        wave, rate = sf.read(path, dtype="float32")
        info = sf.info(path)
        require(rate == RATE and wave.shape == (SAMPLES,) and info.subtype == "PCM_24", "Invalid exported WAV")
        require(np.isfinite(wave).all(), "Non-finite WAV")
        maximum_peak = max(maximum_peak, float(np.max(np.abs(wave))))
        require(waveform_hash(wave) == entry["waveform_sha256"], "WAV sample hash differs")
        require(sha256_file(path) == entry["file_sha256"], "WAV file hash differs")
        checked_images.add(entry["waveform_image"])
    require(maximum_peak <= PEAK_LIMIT + 1e-6, "Unexpected overflow")
    require(len(entries) == len({e["path"] for e in entries}), "Repeated exported paths")
    for relative in assets.paths | checked_images:
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file(), f"Missing/nonlocal resource: {relative}")
        if path.suffix == ".png":
            pixels = np.asarray(Image.open(path).convert("RGB"))
            require(pixels.std() > 1 and min(pixels.shape[:2]) >= 90, f"Blank image: {relative}")
    check_csv(root / "baseline_predictions.csv", baseline)
    check_csv(root / "degradation_predictions.csv", rows)
    result = {"status": "passed", "test_samples": len(baseline), "systems": 30,
              "candidate_sources": 127, "candidate_scores": len(rows), "exported_wavs": len(entries),
              "regenerated_source_groups": 127, "maximum_exported_peak": maximum_peak,
              "full_summary_recomputed": True, "wav_hashes_verified": True,
              "manual_audition": "not_performed"}
    return result, dataset, regenerated, cases, baseline


def rescore_exports(root, cfg, dataset, regenerated, cases, baseline):
    import torch
    from src.runtime import bind_gpu

    gpu = choose_gpu()
    print(json.dumps({"verification_gpu": gpu, "allocator_limit_gib": 6, "batch_limit": 16,
                      "precision": "bf16", "training": False}), flush=True)
    bind_gpu(gpu)
    torch.set_num_threads(4)
    set_seed(42)
    identity = json.loads((root / "run_identity.json").read_text())
    scorer = Scorer(cfg, Path(cfg.storage.output_root) / identity["checkpoint"])
    errors = []
    for group in cases["sources"] + cases["limitations"]:
        clip_id = group["record"]["clip_id"]
        waves, metadata = regenerated[clip_id]
        waves = list(waves)
        for entry in group["variants"]:
            slot = next(i for i, row in enumerate(metadata) if row["variant"] == entry["variant"])
            waves[slot], _ = sf.read(root / entry["path"], dtype="float32")
        scores = scorer(waves)
        for entry in group["variants"]:
            slot = next(i for i, row in enumerate(metadata) if row["variant"] == entry["variant"])
            errors.extend(abs(scores[slot][h] - entry[h]) for h in ("MI", "TA"))
    for pair in cases["pairs"]:
        # Reproduce the original high/low batch order, independently of blind A/B order.
        entries = sorted(pair["entries"], key=lambda e: -baseline[e["record"]["clip_id"]]["MI"])
        scores = scorer([sf.read(root / e["path"], dtype="float32")[0] for e in entries])
        errors.extend(abs(s[h] - e[h]) for s, e in zip(scores, entries) for h in ("MI", "TA"))
    maximum = max(errors)
    require(maximum <= TOLERANCE, f"Final WAV scores not reproducible: {maximum}")
    return {"status": "passed", "gpu": gpu, "maximum_absolute_score_error": maximum,
            "precision": "bf16", "batch_shapes": "13 for degradation groups; 2 for pairs"}


def browser_checks(root):
    from playwright.sync_api import sync_playwright

    target = root / "verification"
    target.mkdir(exist_ok=True)
    errors, external, screenshots = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("request", lambda request: external.append(request.url)
                if urlparse(request.url).scheme not in ("file", "data", "blob") else None)
        page.goto((root / "index.html").as_uri(), wait_until="load")
        require(page.locator("body").evaluate("e=>e.classList.contains('masked')"), "Scores not blind by default")
        require(page.locator("#degradation-tracks .score").first.evaluate(
            "e=>getComputedStyle(e).visibility") == "hidden", "Default score visible")
        require(not page.locator("#source-curve").is_visible(), "Default curve reveals scores")
        require(page.locator("#degradation-tracks audio").count() == 4, "Missing severity players")
        page.locator("#reveal").check()
        require(page.locator("#source-curve").is_visible(), "Score curve not shown")
        for source in range(3):
            page.select_option("#source-picker", str(source))
            for method in METHODS:
                page.locator(f"[data-method='{method}']").click()
                page.wait_for_function("Array.from(document.querySelectorAll('#degradation-tracks audio')).every(a=>a.readyState>=2&&Math.abs(a.duration-10)<.01)")
                require(page.locator("#degradation-tracks audio").count() == 4, "Severity switch broke")
                require(page.locator("#source-curve").evaluate("e=>e.complete&&e.naturalWidth>0"), "Curve missing")
        page.select_option("#source-picker", "0")
        page.locator("[data-method='noise']").click()
        page.wait_for_function("document.querySelector('#degradation-tracks audio').readyState>=2")
        page.locator("#degradation-tracks audio").nth(0).evaluate("async a=>{await a.play();a.currentTime=3;}")
        page.wait_for_timeout(350)
        page.locator("#degradation-tracks audio").nth(1).evaluate("a=>a.play()")
        page.wait_for_timeout(150)
        state = page.locator("#degradation-tracks audio").evaluate_all("aa=>aa.map(a=>({paused:a.paused,time:a.currentTime}))")
        require(state[0]["paused"] and not state[1]["paused"] and 3 <= state[1]["time"] < 4.5,
                f"Same-position switch failed: {state}")
        require(page.locator("audio").evaluate_all("aa=>aa.filter(a=>!a.paused).length") == 1,
                "Multiple tracks playing")
        page.locator("#link-position").uncheck()
        page.locator("#degradation-tracks audio").nth(2).evaluate("a=>a.play()")
        page.wait_for_timeout(100)
        require(page.locator("#degradation-tracks audio").nth(2).evaluate("a=>a.currentTime") < 1,
                "Disabled linking still changed position")
        page.locator("#tab-pairs").click()
        require(page.locator("audio").evaluate_all("aa=>aa.every(a=>a.paused)"), "Tab switch did not stop audio")
        for pair in range(3):
            page.select_option("#pair-picker", str(pair))
            require(page.locator("#pair-tracks .track-title").all_text_contents() == ["片段 A", "片段 B"], "A/B labels changed")
        page.locator("#reveal").uncheck()
        require(page.locator("#pair-tracks .filemeta").first.evaluate("e=>getComputedStyle(e).visibility") == "hidden",
                "Blind pair exposes category")
        page.locator("#reveal").check()
        page.screenshot(path=str(target / "desktop_pairs.png"), full_page=True)
        screenshots.append("verification/desktop_pairs.png")
        page.locator("#tab-limitations").click()
        require(page.locator("#limitation-tracks audio").count() == 2, "Counterexample player missing")
        page.screenshot(path=str(target / "desktop_limitations.png"), full_page=True)
        screenshots.append("verification/desktop_limitations.png")
        page.locator("#tab-degradation").click()
        for name, width, height in (("desktop", 1440, 1000), ("tablet", 768, 1024), ("mobile", 390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(150)
            require(page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"), f"Page overflows on {name}")
            require(page.locator("img").evaluate_all("aa=>aa.every(a=>a.complete&&a.naturalWidth>0)"), "Broken image")
            page.screenshot(path=str(target / f"{name}_degradation.png"), full_page=True)
            screenshots.append(f"verification/{name}_degradation.png")
        page.locator("#reveal").uncheck()
        page.screenshot(path=str(target / "mobile_blind.png"), full_page=True)
        screenshots.append("verification/mobile_blind.png")
        for relative in screenshots:
            pixels = np.asarray(Image.open(root / relative).convert("RGB"))
            require(pixels.std() > 10, f"Blank browser screenshot: {relative}")
        require(not errors, f"Browser errors: {errors}")
        require(not external, f"Unexpected network access: {external}")
        browser.close()
    return {"status": "passed", "engine": "Chromium", "transport": "file://",
            "viewports": ["1440x1000", "768x1024", "390x844"], "screenshots": screenshots,
            "single_audio": True, "position_switching": True, "scores_hidden_by_default": True,
            "external_requests": external, "javascript_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/A1_true_ttm.yaml")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--browser", action="store_true")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    root = Path(args.output_dir).resolve() if args.output_dir else Path(cfg.storage.output_root) / "A1_classroom_degradation"
    result, dataset, regenerated, cases, baseline = audit(root, cfg)
    print("Complete-pool, deterministic waveform, CSV, manifest and exported WAV audit passed", flush=True)
    result["gpu_rescore"] = rescore_exports(root, cfg, dataset, regenerated, cases, baseline) if args.rescore else {"status": "not_run"}
    result["browser"] = browser_checks(root) if args.browser else {"status": "not_run"}
    save_json(root / "verification_results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
