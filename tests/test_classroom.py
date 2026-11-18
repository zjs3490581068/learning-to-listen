"""Fixed interventions, unfiltered summary statistics, and classroom safeguards."""

import copy
import unittest
import os

import numpy as np
from scipy import signal

from src.classroom_degradation import (
    METHODS, PEAK_LIMIT, RATE, SAMPLES, make_variants, pcm24_roundtrip, select_pairs,
    shared_headroom, source_ranking, summarize_interventions, waveform_hash,
)
from scripts.run_classroom_demo import validate_selection


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = np.arange(SAMPLES) / RATE
        cls.wave = (.22 * np.sin(2 * np.pi * 440 * t) + .12 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
        cls.waves, cls.rows = make_variants(cls.wave, "test.wav")

    def test_length_finite_pcm_and_headroom(self):
        self.assertEqual(len(self.waves), 13)
        self.assertEqual(len({r["variant"] for r in self.rows}), 13)
        for wave, row in zip(self.waves, self.rows):
            self.assertEqual(wave.shape, (SAMPLES,))
            self.assertTrue(np.isfinite(wave).all())
            self.assertLessEqual(np.max(np.abs(wave)), PEAK_LIMIT + 1e-6)
            self.assertTrue(np.array_equal(pcm24_roundtrip(wave), wave))
            self.assertEqual(row["waveform_sha256"], waveform_hash(wave))

    def test_exact_snr_and_shared_noise_direction(self):
        clean = self.waves[0].astype(np.float64)
        noises = []
        for i, snr in enumerate((20, 10, 0), 1):
            actual = self.waves[i].astype(np.float64) - clean
            measured = 10 * np.log10(np.mean(clean ** 2) / np.mean(actual ** 2))
            self.assertAlmostEqual(measured, snr, places=3)
            self.assertAlmostEqual(self.rows[i]["diagnostics"]["measured_snr_db"], snr, places=8)
            noises.append(actual / np.sqrt(np.mean(actual ** 2)))
        self.assertGreater(np.corrcoef(noises[0], noises[-1])[0, 1], .999999)

    def test_lowpass_suppresses_high_not_low_frequency(self):
        gain = self.rows[0]["common_gain"]
        t = np.arange(SAMPLES) / RATE
        def amplitude(wave, hz):
            return abs(2 * np.mean(wave * np.exp(-2j * np.pi * hz * t)))
        for i in (4, 5, 6):
            self.assertLess(amplitude(self.waves[i], 6000), .12 * gain * .01)
            self.assertGreater(amplitude(self.waves[i], 440), .22 * gain * .95)

    def test_clipping_thresholds_and_fraction(self):
        fractions = []
        for index in (7, 8, 9):
            row = self.rows[index]
            maximum = row["diagnostics"]["threshold"] * row["common_gain"]
            self.assertLessEqual(np.max(np.abs(self.waves[index])), maximum + 2e-7)
            fractions.append(row["diagnostics"]["clipped_fraction"])
        self.assertTrue(np.all(np.diff(fractions) > 0))

    def test_dropout_nested_disjoint_and_not_time_stretched(self):
        previous = set()
        for index, count in zip((10, 11, 12), (5, 15, 30)):
            intervals = self.rows[index]["diagnostics"]["intervals_samples"]
            current = {start for start, _ in intervals}
            self.assertEqual(len(intervals), count)
            self.assertLessEqual(previous, current)
            for start, end in intervals:
                self.assertEqual(end - start, 2400)
                self.assertTrue(np.all(self.waves[index][start + 120:end - 120] == 0))
            mask = np.ones(SAMPLES, dtype=bool)
            for start, end in intervals:
                mask[start:end] = False
            self.assertTrue(np.array_equal(self.waves[index][mask], self.waves[0][mask]))
            previous = current

    def test_deterministic_seed_and_filename_stream(self):
        same, _ = make_variants(self.wave, "test.wav")
        other, _ = make_variants(self.wave, "other.wav")
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(same, self.waves)))
        # Compare normalized noise because group headroom may depend on its random peaks.
        self.assertLess(np.corrcoef(same[1] - same[0], other[1] - other[0])[0, 1], .05)

    def test_gain_is_common_and_never_amplifies(self):
        waves, gain = shared_headroom([self.wave, self.wave * 8])
        self.assertLess(gain, 1)
        self.assertTrue(np.allclose(waves[0], self.wave * gain))
        self.assertTrue(np.allclose(waves[1], self.wave * 8 * gain))
        quiet, gain = shared_headroom([self.wave * .01])
        self.assertEqual(gain, 1)

    def test_rejects_silent_malformed_and_nonfinite(self):
        for wave in (np.zeros(SAMPLES), np.zeros(100), np.full(SAMPLES, np.nan)):
            with self.assertRaises(ValueError):
                make_variants(wave, "bad.wav")


def synthetic_rows():
    baseline, interventions = [], []
    for index in range(8):
        high = index >= 4
        baseline.append({"clip_id": index, "MI": 4.5 if high else 1.5, "TA": 3.,
                         "mi_score": 4.8 if high else 1.4, "duration": 30.,
                         "prompt_id": f"P{index % 4:03}", "prompt_key": f"text-{index % 4}",
                         "model_id": f"S{index:03}",
                         "audio_filename": f"clip{index}.wav"})
        if not high:
            continue
        interventions.append({"clip_id": index, "variant": "clean", "method": "clean", "severity": 0,
                              "MI": 4.5, "drop_MI": 0., "drop_TA": 0.})
        for method in METHODS:
            for severity in (1, 2, 3):
                drop = severity * (.1 + index * .02)
                if index == 4 and method == "noise":
                    drop = -severity * .1
                interventions.append({"clip_id": index, "variant": f"{method}_{severity}", "method": method,
                                      "severity": severity, "MI": 4.5 - drop,
                                      "drop_MI": drop, "drop_TA": drop / 2})
    return baseline, interventions


class SelectionAndSummaryTests(unittest.TestCase):
    def test_summary_keeps_increases_and_uses_all_sources(self):
        baseline, rows = synthetic_rows()
        summary = summarize_interventions(rows)
        noise = summary["methods"]["noise"]
        self.assertEqual(noise["levels"]["3"]["n"], 4)
        self.assertEqual(noise["levels"]["3"]["increase_count"], 1)
        self.assertEqual(noise["levels"]["3"]["decrease_rate"], .75)
        self.assertEqual(noise["strict_monotonic_rate"], .75)
        self.assertEqual(summary["methods"]["clipping"]["strict_monotonic_rate"], 1)
        ci = noise["levels"]["3"]["wilson_95"]
        self.assertLess(ci[0], .75)
        self.assertGreater(ci[1], .75)

    def test_source_ranking_distinct_reproducible_and_honest(self):
        baseline, rows = synthetic_rows()
        selected, ranked = source_ranking(baseline, rows)
        self.assertEqual(selected, [7, 6, 5])
        self.assertEqual(len({baseline[i]["model_id"] for i in selected}), 3)
        self.assertEqual(len({baseline[i]["prompt_id"] for i in selected}), 3)
        self.assertEqual(ranked[-1]["strict_monotonic_methods"], 3)
        self.assertEqual(source_ranking(baseline, rows)[0], selected)

    def test_same_prompt_pairs_and_no_duplicates(self):
        baseline, _ = synthetic_rows()
        pairs = select_pairs(baseline)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(p["same_prompt"] for p in pairs))
        ids = [p[k] for p in pairs for k in ("high", "low")]
        self.assertEqual(len(set(ids)), 6)
        for pair in pairs:
            self.assertGreater(baseline[pair["high"]]["MI"], baseline[pair["low"]]["MI"])

    def test_selection_overrides_cannot_bypass_quality_gates(self):
        baseline, rows = synthetic_rows()
        sources, ranked = source_ranking(baseline, rows)
        selection = {"sources": sources, "pairs": select_pairs(baseline),
                     "counterexample": {"clip_id": 4, "variant": "noise_3"}}
        validate_selection(selection, ranked, baseline, rows)
        invalid = copy.deepcopy(selection)
        invalid["sources"][0] = 0
        with self.assertRaises(ValueError):
            validate_selection(invalid, ranked, baseline, rows)
        invalid = copy.deepcopy(selection)
        invalid["pairs"][0]["same_prompt"] = False
        with self.assertRaises(ValueError):
            validate_selection(invalid, ranked, baseline, rows)
        invalid = copy.deepcopy(selection)
        invalid["counterexample"]["variant"] = "clipping_3"
        with self.assertRaises(ValueError):
            validate_selection(invalid, ranked, baseline, rows)

    def test_same_id_different_actual_prompts_are_not_same_prompt(self):
        baseline, _ = synthetic_rows()
        for row in baseline[4:]:
            row["prompt_key"] += "-different-actual-text"
        pairs = select_pairs(baseline)
        self.assertFalse(any(p["same_prompt"] for p in pairs))

    def test_manual_sources_cannot_bypass_diversity(self):
        baseline, rows = synthetic_rows()
        sources, ranked = source_ranking(baseline, rows)
        selection = {"sources": sources, "pairs": select_pairs(baseline), "counterexample": None}
        for key in ("model_id", "prompt_key"):
            altered = copy.deepcopy(baseline)
            altered[sources[1]][key] = altered[sources[0]][key]
            with self.assertRaises(ValueError):
                validate_selection(selection, ranked, altered, rows)

    @unittest.skipUnless(os.environ.get("MUQEVAL_INTEGRATION_PATHS"), "Local assets not requested; no downloads")
    def test_actual_prompt_overrides_and_candidate_pool(self):
        from scripts.run_classroom_demo import make_test_manifest
        from train import load_config
        _, manifest = make_test_manifest(load_config("configs/A1_true_ttm.yaml", os.environ["MUQEVAL_INTEGRATION_PATHS"]))
        self.assertEqual(len(manifest["candidate_indices"]), 127)
        self.assertEqual(manifest["n_systems"], 30)
        record = next(r for r in manifest["records"] if r["audio_filename"] == "audiomos2025-track1-S017_P113.wav")
        self.assertEqual(record["prompt_origin"], "per-audio override")
        self.assertIn("rock", record["prompt"])


if __name__ == "__main__":
    unittest.main()
