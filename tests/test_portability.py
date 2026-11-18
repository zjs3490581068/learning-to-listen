"""Release-specific tests that do not require assets, CUDA or networking."""

import copy
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from src.config import ROOT, load_config, scientific_config
from src.data_protocol import make_manifest, validate_manifest
from src.music_profile import class_counts, load_profile, validate_profile
from src.runtime import available_gpus, choose_gpu, source_revision


class PortableTests(unittest.TestCase):
    def test_entry_imports_do_not_change_runtime_environment(self):
        import importlib
        sentinel = {"HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0", "CUDA_VISIBLE_DEVICES": "2", "http_proxy": "http://example.invalid"}
        with patch.dict(os.environ, sentinel):
            for name in ("scripts.run_classroom_demo", "scripts.run_personalization", "scripts.predict_preference", "scripts.audit_local_audio"):
                importlib.reload(importlib.import_module(name))
            self.assertEqual({k: os.environ[k] for k in sentinel}, sentinel)
    def test_custom_preparation_and_full_directory_relocation(self):
        import io
        import numpy as np
        import soundfile as sf
        from datasets import Dataset
        from src.personalization_data import prepare_data, validate_manifest, decode_source
        from src.personalization import identity_for
        profile = load_profile("configs/music.custom.example.yaml")
        profile.update(clips_per_song=2, validation_per_song=1, musiceval_count=4,
                       musiceval_validation=1)
        profile["songs"][0]["protected_interval"] = [40, 50]
        audio = io.BytesIO()
        sf.write(audio, .1 * np.sin(np.arange(240000) * .03), 24000, format="WAV", subtype="PCM_24")
        rows = lambda split: [{"audio": {"path": f"audiomos2025-track1-S00{1 if split == 'train' else 2}_P{i:03}.wav", "bytes": audio.getvalue()},
                              "overall quality": 4., "textual alignment": 4.} for i in range(1, 5 if split == "train" else 2)]
        with tempfile.TemporaryDirectory(prefix="muq-custom-") as folder:
            base = Path(folder)
            music = base / "music"
            music.mkdir()
            wave = (.1 * np.sin(np.arange(110 * 24000) * .03)).astype(np.float32)
            # Unit-test the protocol, not the optional system Vorbis encoder.
            # Real audio decoding has separate tests and opt-in asset checks.
            for i, name in enumerate(("favorite.ogg", "other.ogg")):
                (music / name).write_bytes(f"mock Ogg source {i}".encode())
            profile_path = base / "profile.yaml"
            OmegaConf.save(OmegaConf.create(profile), profile_path)
            cfg = load_config("configs/A3a_custom.example.yaml")
            cfg.personalization.music_manifest = str(profile_path)
            cfg.personalization.music_dir = str(music)
            cfg.paths.output_dir = str(base / "run")
            author = base / "synthetic_author.bin"
            author.write_bytes(b"synthetic identity fixture, not weights")
            cfg.personalization.author_checkpoint = str(author)
            def decode_fixture(source):
                if isinstance(source, (Path, str)):
                    return wave, {"duration_sec": 110., "common_gain": 1., "common_gain_db": 0.}
                return decode_source(source)
            with patch("src.personalization_data.load_dataset", side_effect=lambda path, split: Dataset.from_list(rows(split))), patch("src.personalization_data.check_original_annotations"), patch("src.personalization_data.decode_source", side_effect=decode_fixture):
                manifest = prepare_data(cfg)
                self.assertEqual(manifest, prepare_data(cfg))
            self.assertEqual(len(manifest["samples"]), 8)
            self.assertEqual(len(manifest["demonstrations"]), 2)
            self.assertTrue(all(not Path(s["source_path"]).is_absolute() for s in manifest["personal_sources"]))
            identity = identity_for(cfg, manifest)
            shutil.copytree(base / "run", base / "relocated_run")
            shutil.copytree(music, base / "relocated_music")
            cfg.paths.output_dir = str(base / "relocated_run")
            cfg.personalization.music_dir = str(base / "relocated_music")
            validate_manifest(manifest, cfg.paths.output_dir, music_dir=cfg.personalization.music_dir)
            self.assertEqual(identity, identity_for(cfg, manifest))
            changed = Path(cfg.paths.output_dir) / manifest["samples"][0]["path"]
            changed.write_bytes(b"corrupt test fixture")
            with self.assertRaises(ValueError):
                validate_manifest(manifest, cfg.paths.output_dir, music_dir=cfg.personalization.music_dir)

    def test_relative_paths_and_recursive_inheritance(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                cfg = load_config("configs/A3a_custom.example.yaml")
                self.assertEqual(cfg.model.head_hidden_dim, 512)
                self.assertEqual(cfg.training.warmup_steps, 24)
                self.assertEqual(cfg.paths.output_dir, str(ROOT / "outputs/A3a_personalized_custom"))
            finally:
                os.chdir(old)

    def test_storage_override_and_relocation_identity(self):
        baseline = load_config("configs/A3a_personalized.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.yaml"
            OmegaConf.save(OmegaConf.create({"storage": {"output_root": directory}}), path)
            relocated = load_config("configs/A3a_personalized.yaml", path)
            self.assertNotEqual(baseline.paths.output_dir, relocated.paths.output_dir)
            self.assertEqual(scientific_config(baseline), scientific_config(relocated))
            OmegaConf.save(OmegaConf.create({"training": {"batch_size": 1}}), path)
            with self.assertRaises(ValueError):
                load_config("configs/A1_true_ttm.yaml", path)

    def test_source_export_without_git(self):
        with patch("src.runtime.subprocess.check_output", side_effect=FileNotFoundError):
            self.assertIn("source-export", source_revision())

    def test_visible_gpu_and_uuid_selection(self):
        status = "0, GPU-aaa, 12000\n1, GPU-bbb, 9000\n2, GPU-ccc, 16000\n"
        with patch("src.runtime.subprocess.check_output", return_value=status):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                self.assertEqual(available_gpus(), [(1, 9000)])
                self.assertEqual(available_gpus(requested="2"), [])
                with self.assertRaises(RuntimeError):
                    choose_gpu("2")
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-ccc"}):
                self.assertEqual(available_gpus(), [(2, 16000)])
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                with self.assertRaises(RuntimeError):
                    choose_gpu()

    def test_original_and_custom_profiles(self):
        original = load_profile()
        self.assertEqual(class_counts(original), {"train": {5: 48, 3: 6, 2: 18, 1: 48},
                                                  "validation": {5: 16, 3: 2, 2: 6, 1: 6}})
        custom = load_profile("configs/music.custom.example.yaml")
        self.assertNotEqual(custom["protocol"], original["protocol"])
        self.assertEqual(sum(class_counts(custom)["train"].values()), 22)
        invalid = copy.deepcopy(custom)
        invalid["songs"][0]["pattern"] = "../outside.ogg"
        with self.assertRaises(ValueError):
            validate_profile(invalid)
        with tempfile.TemporaryDirectory() as folder:
            changed = copy.deepcopy(original)
            changed["songs"][0]["preference_target"] = 1
            path = Path(folder) / "changed.yaml"
            OmegaConf.save(OmegaConf.create(changed), path)
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_cache_location_not_dataset_identity(self):
        from tests.test_true_ttm import rows_for_systems
        from src.data_protocol import dataset_records
        rows = rows_for_systems()
        records = dataset_records(rows)
        manifest = make_manifest(rows, records)
        manifest["dataset_fingerprint"] = "old-local-cache-fingerprint"
        validate_manifest(manifest, rows, records, 5, 42)
        manifest["records"][0]["mi_score"] += .1
        with self.assertRaises(ValueError):
            validate_manifest(manifest, rows, records, 5, 42)

    def test_changed_audio_bytes_rejected_with_same_filenames_and_labels(self):
        from tests.test_true_ttm import rows_for_systems
        from src.data_protocol import dataset_records
        rows = rows_for_systems()
        rows[0]["audio"]["bytes"] = b"synthetic original"
        records = dataset_records(rows)
        manifest = make_manifest(rows, records)
        rows[0]["audio"]["bytes"] = b"synthetic replacement"
        self.assertEqual(records, dataset_records(rows))
        with self.assertRaises(ValueError):
            validate_manifest(manifest, rows, records, 5, 42)


if __name__ == "__main__":
    unittest.main()
