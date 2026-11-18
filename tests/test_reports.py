"""Generate both offline pages from synthetic audio, never copyrighted examples."""

import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import zipfile

import numpy as np
import soundfile as sf

from src.classroom_degradation import METHODS, RATE, SAMPLES, make_variants, summarize_interventions
from src.classroom_report import build_report as classroom_report
from src.personalization import metrics
from src.personalization_data import save_json
from src.personalization_report import build_report as personal_report
from tests.test_classroom import synthetic_rows


def save_wave(root, path, wave):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, wave, RATE, subtype="PCM_24")


def classroom_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    baseline, rows = synthetic_rows()
    for record in baseline:
        record.update(prompt="Synthetic music example", ta_score=3.)
    payload = {"sources": [], "pairs": [], "limitations": [], "methods": METHODS}
    for clip_id in (7, 6, 5, 4):
        wave = (.2 * np.sin(2 * np.pi * (220 + clip_id * 30) * np.arange(SAMPLES) / RATE)).astype(np.float32)
        waves, metadata = make_variants(wave, f"clip{clip_id}.wav")
        variants = []
        for wave, meta in zip(waves, metadata):
            score = next(r for r in rows if r["clip_id"] == clip_id and r["variant"] == meta["variant"])
            path = f"audio/source_{clip_id}/{meta['variant']}.wav"
            save_wave(root, path, wave)
            variants.append({**meta, **score, "TA": 3. - score["drop_TA"], "path": path})
        group = {"record": baseline[clip_id], "variants": variants, "selection": "Synthetic UI test only"}
        if clip_id == 4:
            group.update(variants=variants[:2], increase=.1)
            payload["limitations"].append(group)
        else:
            payload["sources"].append(group)
    for number in range(3):
        entries = []
        for letter, index in (("A", number), ("B", number + 4)):
            path = f"audio/pair_{number}/{letter}.wav"
            save_wave(root, path, waves[0])
            entries.append({"path": path, "letter": letter, "MI": baseline[index]["MI"], "TA": 3.,
                            "record": baseline[index], "category": "synthetic"})
        payload["pairs"].append({"name": f"Pair {number + 1}", "same_prompt": True, "common_gain": 1., "entries": entries})
    summary = summarize_interventions(rows)
    summary.update(n_baseline=len(baseline), n_sources=4, n_systems=8, wall_time_sec=0.,
                   peak_allocated_gib=0., peak_reserved_gib=0.)
    classroom_report(root, payload, summary, baseline, rows)
    for name in ("baseline_predictions.csv", "degradation_predictions.csv"):
        (root / name).write_text("synthetic_test_only\n")
    save_json(root / "selection.json", {"note": "synthetic test only"})


def personal_fixture(root, demo_count=5):
    root.mkdir(parents=True, exist_ok=True)
    wave = (.2 * np.sin(2 * np.pi * 330 * np.arange(SAMPLES) / RATE)).astype(np.float32)
    demos = []
    for index, target in enumerate((5, 5, 3, 2, 1)[:demo_count]):
        path = f"audio/demo/synthetic_{index}.wav"
        save_wave(root, path, wave)
        demos.append({"sample_id": f"demo_{index}", "split": "demo", "path": path, "title": f"Synthetic {index}",
                      "source_kind": "personal", "artist": "Synthetic", "start_sec": 90., "end_sec": 100.,
                      "preference_target": target, "author_MI_score": 3., "preference_score": float(target),
                      "display_target_interval": [max(1, target - .5), min(5, target + .5)], "display_target_met": True})
    samples = [{"sample_id": f"{split}_{target}", "preference_target": target, "split": split,
                "author_MI_score": 3., "preference_score": float(target)}
               for split in ("train", "validation") for target in (1, 2, 3, 5)]
    result = metrics([1., 2., 3., 5.], [1., 2., 3., 5.])
    evaluation = {"demonstrations": demos, "metrics": {"validation": result},
                  "author_baseline_metrics": {"validation": metrics([1., 2., 3., 5.], [3., 3., 3., 3.])},
                  "display_targets_met": len(demos)}
    data = {"evaluation_results.json": evaluation,
            "training_results.json": {"best_epoch": 2, "epochs_run": 2},
            "dataset_manifest.json": {"protocol": "synthetic_ui_only", "personal_sources": [{}, {}], "samples": samples},
            "predictions.json": samples + demos,
            "training_history.json": [{"epoch": e, "train_loss": 1. / e, "val_loss": 1.1 / e,
                                       "val_macro_mae": .5 / e, "val_mae": .4 / e} for e in (1, 2)],
            "run_status.json": {"status": "synthetic_test_only"}}
    for name, value in data.items():
        save_json(root / name, value)
    for name in ("demonstration_scores.csv", "predictions.csv", "clip_manifest.csv"):
        (root / name).write_text("synthetic_test_only\n")
    personal_report(root)


class ReportTests(unittest.TestCase):
    def test_classroom_generation(self):
        with tempfile.TemporaryDirectory(prefix="muq-web-") as folder:
            root = Path(folder)
            (root / "README.md").write_text("Maintained documentation")
            classroom_fixture(root)
            self.assertEqual((root / "README.md").read_text(), "Maintained documentation")
            self.assertTrue((root / "REPORT.md").is_file())
            html = (root / "index.html").read_text()
            self.assertNotIn("__CLASSROOM_DATA__", html)
            self.assertIn('id="pool-count"', html)
            script = Path(__file__).resolve().parents[1] / "scripts/package_demo.py"
            subprocess.run([sys.executable, str(script), "--kind", "a1", "--output-dir", str(root)], check=True, capture_output=True)
            with zipfile.ZipFile(root / "classroom.zip") as archive:
                self.assertEqual(sum(n.endswith(".wav") for n in archive.namelist()), 47)
                self.assertIn("selection.json", archive.namelist())

    def test_personal_generation(self):
        with tempfile.TemporaryDirectory(prefix="muq-web-") as folder:
            root = Path(folder)
            personal_fixture(root)
            payload = json.loads((root / "presentation.json").read_text())
            self.assertEqual(payload["counts"], {"train": 4, "validation": 4})
            self.assertNotIn("__PERSONAL_DATA__", (root / "index.html").read_text())
            self.assertTrue((root / "REPORT.md").is_file())
            save_wave(root, "audio/train/not_for_presentation.wav", np.zeros(SAMPLES, dtype=np.float32))
            script = Path(__file__).resolve().parents[1] / "scripts/package_demo.py"
            subprocess.run([sys.executable, str(script), "--kind", "personal", "--output-dir", str(root)], check=True, capture_output=True)
            with zipfile.ZipFile(root / "classroom.zip") as archive:
                self.assertEqual(sum(n.endswith(".wav") for n in archive.namelist()), 5)
                self.assertFalse(any("audio/train" in n for n in archive.namelist()))

    @unittest.skipUnless(os.environ.get("MUQEVAL_BROWSER_TESTS"), "Optional installed browser not requested")
    def test_offline_browser_both_pages(self):
        from scripts.verify_classroom_demo import browser_checks
        from scripts.verify_personalization import browser_audit
        with tempfile.TemporaryDirectory(prefix="muq-browser-") as folder:
            root = Path(folder)
            classroom_fixture(root / "a1")
            personal_fixture(root / "personal", demo_count=2)
            self.assertEqual(browser_checks(root / "a1")["status"], "passed")
            self.assertEqual(browser_audit(root / "personal")["status"], "passed")


if __name__ == "__main__":
    unittest.main()
