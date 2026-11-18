from collections import Counter
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import os

import numpy as np
from omegaconf import OmegaConf
import soundfile as sf
import torch

from src.data import AudioProcessor, MusicEvalDataset
from src.data_protocol import (
    PROTOCOL, check_original_annotations, dataset_records, make_manifest,
    parse_system_id, records_fingerprint, stratified_splits, validate_manifest,
)
from src.evaluation import bootstrap_ci, full_evaluation
from src.inference import collect_predictions, load_checkpoint
from src.experiment_results import check_completed_fold


def rows_for_systems():
    rows = []
    for system in range(1, 34):
        if system in (5, 25):
            continue
        for prompt in range(1, 7):
            rows.append({"audio": {"path": f"audiomos2025-track1-S{system:03}_P{prompt:03}.wav"},
                         "overall quality": 1 + prompt % 5,
                         "textual alignment": 1 + system % 5})
    return rows


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.dataset = rows_for_systems()
        self.records = dataset_records(self.dataset)

    def test_true_id_parsing(self):
        self.assertEqual(parse_system_id("/arbitrary/audiomos2025-track1-S002_P044.wav"), "S002")
        self.assertEqual(parse_system_id("audiomos2025-track1-S033_P001.wav"), "S033")

    def test_missing_or_invalid_ids_fail(self):
        for value in (None, "", "clip.wav", "S001_P001.wav", "audiomos2025-track1-S005_P001.wav",
                      "audiomos2025-track1-S000_P001.wav", "audiomos2025-track1-S034_P001.wav"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_system_id(value)

    def test_duplicates_and_invalid_scores_fail(self):
        with self.assertRaises(ValueError):
            dataset_records(self.dataset + [self.dataset[0]])
        invalid = copy.deepcopy(self.dataset)
        invalid[0]["overall quality"] = float("nan")
        with self.assertRaises(ValueError):
            dataset_records(invalid)

    def test_stratification_coverage_and_reproducibility(self):
        splits = stratified_splits(self.records)
        self.assertEqual(splits, stratified_splits(self.records))
        self.assertNotEqual(splits, stratified_splits(self.records, seed=43))
        coverage = []
        counts = []
        for train, val in splits:
            self.assertFalse(set(train) & set(val))
            self.assertEqual(set(train) | set(val), set(range(len(self.records))))
            coverage.extend(val)
            counts.append(Counter(self.records[i]["model_id"] for i in val))
            self.assertEqual(len(counts[-1]), 31)
        self.assertEqual(sorted(coverage), list(range(len(self.records))))
        for system in counts[0]:
            values = [count[system] for count in counts]
            self.assertLessEqual(max(values) - min(values), 1)

    def test_manifest_rejects_changed_seed_data_or_indices(self):
        manifest = make_manifest(self.dataset, self.records)
        self.assertEqual(validate_manifest(manifest, self.dataset, self.records, 5, 42),
                         stratified_splits(self.records))
        for field, value in [("seed", 43), ("records_sha256", "wrong")]:
            invalid = copy.deepcopy(manifest)
            invalid[field] = value
            with self.assertRaises(ValueError):
                validate_manifest(invalid, self.dataset, self.records, 5, 42)
        invalid = copy.deepcopy(manifest)
        invalid["folds"][0]["val_indices"].reverse()
        with self.assertRaises(ValueError):
            validate_manifest(invalid, self.dataset, self.records, 5, 42)

    def test_original_annotations_must_match(self):
        text = "".join(f"{r['audio_filename']},{r['mi_score']},{r['ta_score']}\n" for r in self.records)
        with patch.object(Path, "open", return_value=io.StringIO(text)):
            check_original_annotations(self.records, "unused")
        changed = copy.deepcopy(self.records)
        changed[0]["mi_score"] = 4.9
        with patch.object(Path, "open", return_value=io.StringIO(text)), self.assertRaises(ValueError):
            check_original_annotations(changed, "unused")

    def test_dataset_item_keeps_identity(self):
        audio = io.BytesIO()
        sf.write(audio, np.ones(1600, dtype=np.float32) * 0.1, 16000, format="WAV")
        row = copy.deepcopy(self.dataset[0])
        row["audio"]["bytes"] = audio.getvalue()
        item = MusicEvalDataset([0], [row], AudioProcessor(), "center")[0]
        self.assertEqual(item["clip_id"], 0)
        self.assertEqual(item["model_id"], "S001")
        self.assertEqual(item["audio_filename"], row["audio"]["path"])
        self.assertEqual(tuple(item["waveform"].shape), (240000,))

    def test_checkpoint_protocol_and_fold_are_checked(self):
        manifest = make_manifest(self.dataset, self.records)
        model = torch.nn.Linear(2, 1)
        checkpoint = {"model_state": model.state_dict(), "data_protocol": {
            "protocol": PROTOCOL, "records_sha256": manifest["records_sha256"],
            "manifest_sha256": records_fingerprint(manifest), "fold": 0}}
        cfg = OmegaConf.create({"data": {"split_manifest": "ignored.json"}})
        with patch("torch.load", return_value=checkpoint), patch.object(Path, "read_text", return_value=json.dumps(manifest)):
            load_checkpoint(model, "ignored.pt", cfg, 0)
            with self.assertRaises(ValueError):
                load_checkpoint(model, "ignored.pt", cfg, 1)
            checkpoint["data_protocol"]["records_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                load_checkpoint(model, "ignored.pt", cfg, 0)

    def test_prediction_identity_order_and_float_conversion(self):
        class Model(torch.nn.Module):
            def forward(self, waveform, dataset_ids):
                return {"MI": waveform[:, 0].to(torch.bfloat16),
                        "TA": waveform[:, 0].to(torch.bfloat16)}

        class Loader:
            class dataset:
                indices = [2, 0, 1]

            def __iter__(self):
                yield {"waveform": torch.tensor([[2.0], [3.0], [4.0]]),
                       "dataset_id": torch.zeros(3, dtype=torch.long),
                       "clip_id": torch.tensor([2, 0, 1]), "model_id": ["S001", "S002", "S003"],
                       "audio_filename": ["a", "b", "c"], "mi_score": torch.tensor([2., 3., 4.]),
                       "ta_score": torch.tensor([2., 3., 4.])}
        cfg = OmegaConf.create({"training": {"mixed_precision": "bf16"},
                               "experiment": {"name": "test"}, "data": {}})
        result = collect_predictions(Model(), Loader(), cfg, 0, "cpu")
        self.assertEqual(result["clip_ids"], [2, 0, 1])
        self.assertEqual(result["predictions_MI"], [2., 3., 4.])
        self.assertEqual(result["model_ids"], ["S001", "S002", "S003"])

    def test_bootstrap_perfect_correlation_is_finite(self):
        from scipy.stats import spearmanr
        values = np.arange(10, dtype=float)
        point, low, high = bootstrap_ci(values, values, lambda p, t: spearmanr(p, t)[0], 100)
        self.assertTrue(np.isfinite([point, low, high]).all())
        self.assertAlmostEqual(point, 1)
        self.assertLessEqual(low, high)

    def test_system_metrics_use_true_grouping(self):
        pred = np.array([1., 1.2, 3., 3.1, 4.8, 5.])
        targets = pred + np.array([0, .05, -.05, .05, -.05, 0])
        result = full_evaluation(pred, targets, np.repeat(["S001", "S002", "S003"], 2), n_bootstrap=0)
        self.assertEqual(result["system"]["n_systems"], 3)
        self.assertAlmostEqual(result["system"]["srcc_sys"], 1)


@unittest.skipUnless(os.environ.get("MUQEVAL_INTEGRATION_PATHS"), "Local assets not requested; no downloads")
class RealDatasetTests(unittest.TestCase):
    def test_official_train_ids_and_folds(self):
        from datasets import Audio, load_dataset
        from train import load_config
        cfg = load_config("configs/A1_true_ttm.yaml", os.environ["MUQEVAL_INTEGRATION_PATHS"])
        ds = load_dataset(cfg.data.musiceval_id, split="train").cast_column("audio", Audio(decode=False))
        records = dataset_records(ds)
        check_original_annotations(records, cfg.data.musiceval_metadata_dir)
        self.assertEqual(len(records), 1923)
        self.assertEqual(len({r["model_id"] for r in records}), 31)
        splits = stratified_splits(records)
        self.assertEqual([len(val) for _, val in splits], [385, 385, 385, 384, 384])
        for _, val in splits:
            self.assertEqual(len({records[i]["model_id"] for i in val}), 31)
        original = load_config("configs/A1_local.yaml", os.environ["MUQEVAL_INTEGRATION_PATHS"])
        self.assertEqual(cfg.model, original.model)
        self.assertEqual(cfg.training, original.training)
        self.assertEqual(cfg.loss, original.loss)


class SchedulerTests(unittest.TestCase):
    def test_gpu_eligibility_and_exclusions(self):
        from scripts.run_a1_true_ttm import available_gpus
        status = "0, GPU-a, 8191\n1, GPU-b, 8192\n2, GPU-c, 12000\n3, GPU-d, 12000\n4, GPU-e, 4000\n"
        with patch.dict(os.environ, {}, clear=True), patch("subprocess.check_output", return_value=status):
            self.assertEqual(available_gpus(), [(2, 12000), (3, 12000), (1, 8192)])
            self.assertEqual(available_gpus({2}), [(3, 12000), (1, 8192)])

    def test_completed_fold_gate_rejects_incomplete_or_nonfinite_results(self):
        training = {"gradient_check": {"frozen_encoder_has_gradients": False,
                                       "pooling_has_gradients": True,
                                       "heads_have_gradients": True},
                    "train_history": [{"epoch": 0, "val/MI_srcc": 0.8}]}
        predictions = {"clip_ids": [0, 2, 4]}
        evaluation = {"n_systems": 31, "n_samples": 3,
                      "heads": {name: {group: {"srcc": 0.8}
                                       for group in ("utterance", "system")}
                                for name in ("MI", "TA")}}
        manifest = {"folds": [{"val_indices": [0, 2, 4]}]}

        def read_json(path):
            return json.dumps({"training_results.json": training,
                               "test_predictions.json": predictions,
                               "evaluation_results.json": evaluation}[path.name])

        with patch.object(Path, "read_text", read_json), patch.object(Path, "is_file", return_value=True):
            check_completed_fold("unused", 0, manifest)
            predictions["clip_ids"].reverse()
            with self.assertRaises(ValueError):
                check_completed_fold("unused", 0, manifest)
            predictions["clip_ids"].reverse()
            training["gradient_check"]["frozen_encoder_has_gradients"] = True
            with self.assertRaises(ValueError):
                check_completed_fold("unused", 0, manifest)
            training["gradient_check"]["frozen_encoder_has_gradients"] = False
            evaluation["heads"]["TA"]["system"]["srcc"] = float("nan")
            with self.assertRaises(ValueError):
                check_completed_fold("unused", 0, manifest)


if __name__ == "__main__":
    unittest.main()
