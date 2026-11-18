"""Deterministic cutting, fixed labels, leakage guards, and preference math."""

import copy
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from src.personalization import freeze_batchnorm, metrics, predict, preference_score
from src.personalization_data import (
    PEAK_LIMIT, PROTOCOL, RATE, PreferenceDataset,
    choose_intervals, export_clip, overlaps, rng, validate_manifest,
)
from src.music_profile import load_profile

_PROFILE = load_profile()
SONGS = {s['title']: (s['preference_target'], s.get('protected_interval')) for s in _PROFILE['songs']}
DEMO_ORDER = _PROFILE['demo_order']


def manifest_fixture():
    sources, samples, demos = [], [], {}
    for i, title in enumerate(SONGS):
        score, protected = SONGS[title]
        source = {"source_id": f"song_{i}", "source_sha256": f"{i:064x}", "title": title,
                  "common_gain": 1., "duration_sec": 500.,
                  "selection": {"head_excluded_sec": 20, "tail_excluded_sec": 10}}
        sources.append(source)
        for slot in range(8):
            samples.append({"sample_id": f"song_{i}_{slot}", "source_id": source["source_id"],
                "source_kind": "personal", "split": "train" if slot < 6 else "validation",
                "start_sec": 120 + 10 * slot, "end_sec": 130 + 10 * slot,
                "preference_target": score, "common_gain": 1., "quiet_fraction": 0., "rms_dbfs": -20.})
        if protected:
            demos[title] = {"sample_id": f"demo_{i}", "title": title, "split": "demo",
                            "start_sec": protected[0], "end_sec": protected[1], "preference_target": score}
    for i in range(54):
        samples.append({"sample_id": f"me_{i}", "source_kind": "musiceval", "source_id": f"me_{i}",
                        "split": "train" if i < 48 else "validation", "preference_target": 1,
                        "official_split": "train", "audio_filename": f"me_{i}.wav"})
    demos["MusicEval"] = {"sample_id": "demo_me", "title": "MusicEval", "split": "demo",
                         "official_split": "test", "audio_filename": "test.wav", "preference_target": 1}
    return {"protocol": PROTOCOL, "personal_sources": sources, "samples": samples,
            "demonstrations": [demos[title] for title in DEMO_ORDER]}


class PersonalCutTests(unittest.TestCase):
    def test_exact_eight_nonoverlap_and_short_song_edges(self):
        wave = np.full(103 * RATE, .2, dtype=np.float32)
        rows, meta = choose_intervals(wave)
        self.assertEqual(len(rows), 8)
        self.assertEqual(meta["head_excluded_sec"], 10)
        for row in rows:
            self.assertGreaterEqual(row["start_sec"], 10)
            self.assertLessEqual(row["end_sec"], 93)
        self.assertTrue(all(a["end_sec"] <= b["start_sec"] for a, b in zip(rows, rows[1:])))

    def test_silence_protected_interval_and_reproducibility(self):
        wave = np.full(200 * RATE, .2, dtype=np.float32)
        wave[40 * RATE:55 * RATE] = 0
        rows, meta = choose_intervals(wave, [90, 100])
        self.assertEqual((rows, meta), choose_intervals(wave, [90, 100]))
        for row in rows:
            self.assertFalse(overlaps((row["start_sec"], row["end_sec"]), (90, 100)))
            self.assertLessEqual(row["quiet_fraction"], .1)
            self.assertGreaterEqual(row["rms_dbfs"], -40)
        self.assertEqual(meta["head_excluded_sec"], 20)

    def test_no_silent_or_insufficient_duration_fallback(self):
        for wave in (np.zeros(120 * RATE, dtype=np.float32), np.full(60 * RATE, .2, dtype=np.float32)):
            with self.assertRaises(ValueError):
                choose_intervals(wave)

    def test_half_open_boundary_and_deterministic_random_streams(self):
        self.assertFalse(overlaps((80, 90), (90, 100)))
        self.assertTrue(overlaps((81, 91), (90, 100)))
        self.assertTrue(np.array_equal(rng("same").permutation(54), rng("same").permutation(54)))
        self.assertFalse(np.array_equal(rng("same").permutation(54), rng("other").permutation(54)))

    def test_pcm_export_and_dataset_tensor_contract(self):
        with tempfile.TemporaryDirectory(prefix="personal-test-") as folder:
            root = Path(folder)
            row = {"sample_id": "opaque", "split": "train", "common_gain": .5, "preference_target": 5}
            exported = export_clip(root, np.full(240000, 1.2, dtype=np.float32), row, 0)
            item = PreferenceDataset(root, [exported])[0]
            self.assertEqual(set(item), {"waveform", "target", "sample_id"})
            self.assertEqual(tuple(item["waveform"].shape), (240000,))
            self.assertEqual(item["waveform"].dtype, torch.float32)
            self.assertLessEqual(exported["peak"], PEAK_LIMIT)
            self.assertEqual(sf.info(root / exported["path"]).subtype, "PCM_24")


class PersonalManifestTests(unittest.TestCase):
    def test_fixed_counts_labels_and_demo_identities(self):
        self.assertTrue(validate_manifest(manifest_fixture(), verify_sources=False))

    def test_demo_leakage_rejected(self):
        manifest = manifest_fixture()
        source_id = next(s["source_id"] for s in manifest["personal_sources"] if s["title"] == "第六单元")
        row = next(r for r in manifest["samples"] if r["source_id"] == source_id)
        row.update(start_sec=90, end_sec=100)
        with self.assertRaises(ValueError):
            validate_manifest(manifest, verify_sources=False)

    def test_count_overlap_label_and_demo_tampering_rejected(self):
        original = manifest_fixture()
        for kind in ("count", "overlap", "label", "demo"):
            manifest = copy.deepcopy(original)
            if kind == "count":
                manifest["samples"].pop()
            elif kind == "overlap":
                manifest["samples"][1].update(start_sec=125, end_sec=135)
            elif kind == "label":
                manifest["samples"][0]["preference_target"] = 2
            else:
                manifest["demonstrations"][0]["start_sec"] = 89
            with self.assertRaises(ValueError, msg=kind):
                validate_manifest(manifest, verify_sources=False)


class PersonalTrainingTests(unittest.TestCase):
    def test_validation_cache_remains_usable_for_training(self):
        class CachedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 5)
                self.cache = None

            def forward(self, x):
                if self.cache is None:
                    self.cache = torch.ones(5)
                return {"MI": self.head(x) * self.cache}

        class SmallDataset(torch.utils.data.Dataset):
            records = [{"sample_id": "sample"}]

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"waveform": torch.ones(3), "target": 5., "sample_id": "sample"}

        model = CachedModel()
        loader = torch.utils.data.DataLoader(SmallDataset())
        with patch.object(torch.Tensor, "cuda", lambda tensor, **kwargs: tensor), \
                patch("src.personalization.torch.autocast", lambda *args, **kwargs: nullcontext()):
            predict(model, loader)
        self.assertFalse(model.cache.is_inference())
        model.train()
        model(torch.ones(1, 3))["MI"].sum().backward()
        self.assertIsNotNone(model.head.weight.grad)

    def test_ordinal_expected_score_not_manual_clipping(self):
        logits = torch.tensor([[10., 0, 0, 0, 0], [0., 0, 10, 0, 0], [0., 0, 0, 0, 10]])
        scores = preference_score(logits)
        self.assertTrue(torch.all(scores >= 1) and torch.all(scores <= 5))
        self.assertAlmostEqual(float(scores[1]), 3., places=5)
        self.assertLess(float(scores[0]), 1.01)
        self.assertGreater(float(scores[2]), 4.99)
        with self.assertRaises(FloatingPointError):
            preference_score(torch.full((2, 5), float("nan")))

    def test_macro_mae_does_not_ignore_small_classes(self):
        result = metrics([1., 1., 1., 3.], [1., 1., 1., 1.])
        self.assertEqual(result["mae"], .5)
        self.assertEqual(result["macro_mae"], 1.)

    def test_frozen_batchnorm_statistics(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(.1)).train()
        before = model[0].running_mean.clone()
        freeze_batchnorm(model)
        model(torch.randn(10, 4))
        self.assertTrue(torch.equal(before, model[0].running_mean))
        self.assertTrue(model[1].training)


if __name__ == "__main__":
    unittest.main()
