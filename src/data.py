"""
Data loading for the official MusicEval splits with MI and TA ratings.

All audio is resampled to 24 kHz and cropped/padded to a fixed duration
for the MuQ encoder input.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import io
import soundfile as sf
from datasets import Audio, load_dataset
from torch.utils.data import Dataset, DataLoader

from .data_protocol import (
    dataset_records, parse_system_id, stratified_splits, validate_manifest,
)


class AudioProcessor:
    """Resample and crop/pad audio to fixed length at target sample rate."""

    def __init__(self, target_sr: int = 24000, clip_samples: int = 240000):
        self.target_sr = target_sr
        self.clip_samples = clip_samples

    def process(self, waveform: np.ndarray, source_sr: int,
                mode: str = "random") -> torch.Tensor:
        """Process raw audio array to fixed-length tensor.

        Args:
            waveform: 1D numpy array of audio samples.
            source_sr: Source sample rate.
            mode: "random" for random crop, "center" for center crop.

        Returns:
            Tensor of shape [clip_samples] at target_sr.
        """
        wav = torch.from_numpy(waveform).float()
        if wav.dim() > 1:
            wav = wav.mean(dim=0)

        # Resample if needed
        if source_sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, source_sr, self.target_sr)

        # Crop or pad
        if wav.shape[0] >= self.clip_samples:
            if mode == "random":
                start = random.randint(0, wav.shape[0] - self.clip_samples)
            else:
                start = (wav.shape[0] - self.clip_samples) // 2
            wav = wav[start:start + self.clip_samples]
        else:
            pad_len = self.clip_samples - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        return wav


class MusicEvalDataset(Dataset):
    """MusicEval dataset with per-clip quality ratings.

    Each sample provides:
      - waveform: [clip_samples] tensor at 24 kHz
      - mi_score: Musical Impression (overall_quality), float 1-5
      - ta_score: Textual Alignment, float 1-5
      - model_id: true TTM system ID from the audio filename
      - audio_filename: original filename, retained for protocol audits
      - clip_id: unique identifier
    """

    def __init__(
        self,
        indices: list[int],
        hf_dataset,
        processor: AudioProcessor,
        crop_mode: str = "random",
    ):
        self.indices = indices
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        item = self.hf_dataset[i]

        # Decode audio with soundfile (avoids torchcodec dependency)
        audio_data = item["audio"]
        if audio_data.get("bytes") is not None:
            array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        else:
            array, sr = sf.read(audio_data["path"])
        if array.ndim > 1:
            array = array.mean(axis=1)  # mono

        waveform = self.processor.process(
            np.array(array, dtype=np.float32),
            sr,
            mode=self.crop_mode,
        )

        return {
            "waveform": waveform,
            "mi_score": torch.tensor(item.get("overall_quality", item.get("overall quality")), dtype=torch.float32),
            "ta_score": torch.tensor(item.get("textual_alignment", item.get("textual alignment")), dtype=torch.float32),
            "dataset_id": 0,  # 0 = MusicEval
            "clip_id": i,
            "model_id": parse_system_id(audio_data.get("path")),
            "audio_filename": Path(audio_data["path"]).name,
        }


def get_musiceval_cv_splits(
    hf_dataset,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Create stratified k-fold splits for MusicEval, stratified by TTM model.

    Returns list of (train_indices, test_indices) tuples.
    """
    return stratified_splits(dataset_records(hf_dataset), n_folds, seed)


def build_dataloaders(
    cfg,
    fold_idx: int = 0,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train and validation dataloaders for a given CV fold.

    Args:
        cfg: OmegaConf config.
        fold_idx: Which CV fold to use (0-indexed).

    Returns:
        (train_loader, val_loader, songeval_loader_or_None)
    """
    processor = AudioProcessor(
        target_sr=cfg.data.sample_rate,
        clip_samples=cfg.data.clip_samples,
    )

    # Load MusicEval (disable auto audio decoding to avoid torchcodec dependency)
    me_dataset = load_dataset(cfg.data.musiceval_id, split="train")
    me_dataset = me_dataset.cast_column("audio", Audio(decode=False))
    records = dataset_records(me_dataset)
    manifest_path = cfg.data.get("split_manifest")
    if manifest_path:
        import json
        with Path(manifest_path).open() as stream:
            manifest = json.load(stream)
        splits = validate_manifest(manifest, me_dataset, records, cfg.data.cv_folds, cfg.seed)
    else:
        splits = stratified_splits(records, cfg.data.cv_folds, cfg.seed)
    if not 0 <= fold_idx < len(splits):
        raise ValueError(f"Invalid fold index: {fold_idx}")
    train_indices, val_indices = splits[fold_idx]

    train_ds = MusicEvalDataset(train_indices, me_dataset, processor, crop_mode="random")
    val_ds = MusicEvalDataset(val_indices, me_dataset, processor, crop_mode="center")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    return train_loader, val_loader, None
