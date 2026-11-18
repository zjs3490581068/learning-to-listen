"""Local audit must validate actual audio bytes, not filename suffixes."""

import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from scripts.audit_local_audio import audit_file, checked_path


class LocalAudioAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.folder = tempfile.TemporaryDirectory(prefix="audio-audit-")
        cls.root = Path(cls.folder.name)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_stereo_resample_center_and_seeded_random_are_reproducible(self):
        t = np.arange(12 * 44100) / 44100
        wave = np.stack([.2 * np.sin(2 * np.pi * 220 * t), .1 * np.sin(2 * np.pi * 440 * t)], axis=1)
        path = self.root / "stereo.wav"
        sf.write(path, wave, 44100)
        state = random.getstate()
        row, inputs = audit_file(path)
        again, repeated = audit_file(path)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(row["status"], "usable")
        self.assertEqual(row["decoded_frames"], len(wave))
        self.assertEqual(row["padding_samples"], 0)
        self.assertEqual(inputs["center"].shape, (240000,))
        self.assertTrue(torch.equal(inputs["random"], repeated["random"]))
        self.assertEqual(row["center_sha256"], again["center_sha256"])

    def test_short_real_flac_is_padded(self):
        path = self.root / "short.flac"
        sf.write(path, .1 * np.sin(np.arange(24000) * .05), 24000)
        row, inputs = audit_file(path)
        self.assertEqual(row["status"], "usable")
        self.assertEqual(row["padding_samples"], 216000)
        self.assertTrue(torch.all(inputs["center"][24000:] == 0))

    def test_fake_flac_rejected_without_modification(self):
        path = self.root / "not_audio.mgg2.flac"
        content = b"Not a real FLAC audio file"
        path.write_bytes(content)
        row, inputs = audit_file(path)
        self.assertEqual(row["status"], "rejected")
        self.assertIsNone(inputs)
        self.assertIn("diagnosis", row)
        self.assertEqual(path.read_bytes(), content)

    def test_silent_audio_rejected(self):
        path = self.root / "silence.wav"
        sf.write(path, np.zeros(24000), 24000)
        row, _ = audit_file(path)
        self.assertEqual(row["status"], "rejected")
        self.assertIn("silent", row["error"])

    def test_portable_path_resolution(self):
        self.assertEqual(checked_path(self.root / "music.wav"), (self.root / "music.wav").resolve())


if __name__ == "__main__":
    unittest.main()
