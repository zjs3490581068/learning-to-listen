"""Validated MusicEval identities and reproducible, persisted CV splits."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import StratifiedKFold


PROTOCOL = "true_ttm_stratified_v1"
FILENAME = re.compile(r"^audiomos2025-track1-(S\d{3})_P\d{3}\.wav$")


def parse_system_id(filename):
    if not isinstance(filename, str) or not filename:
        raise ValueError("MusicEval audio filename is missing")
    name = Path(filename).name
    match = FILENAME.fullmatch(name)
    if match is None or not 1 <= int(match[1][1:]) <= 33 or match[1] in {"S005", "S025"}:
        raise ValueError(f"Invalid MusicEval system/filename: {filename!r}")
    return match[1]


def dataset_records(dataset):
    records = []
    seen = set()
    for index, row in enumerate(dataset):
        path = row["audio"].get("path")
        system_id = parse_system_id(path)
        filename = Path(path).name
        if filename in seen:
            raise ValueError(f"Duplicate MusicEval filename: {filename}")
        seen.add(filename)
        mi = float(row.get("overall_quality", row.get("overall quality")))
        ta = float(row.get("textual_alignment", row.get("textual alignment")))
        if not all(np.isfinite(v) and 1 <= v <= 5 for v in (mi, ta)):
            raise ValueError(f"Invalid MOS for {filename}")
        records.append({"clip_id": index, "audio_filename": filename,
                        "model_id": system_id, "mi_score": mi, "ta_score": ta})
    return records


def records_fingerprint(records):
    blob = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def stratified_splits(records, n_folds=5, seed=42):
    labels = [row["model_id"] for row in records]
    counts = Counter(labels)
    if not counts or min(counts.values()) < n_folds:
        raise ValueError("Each true TTM system must have at least n_folds samples")
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(train.tolist(), val.tolist())
            for train, val in splitter.split(np.arange(len(records)), labels)]


def check_original_annotations(records, metadata_dir, split="train"):
    split_name = {"train": "train", "validation": "dev", "test": "test"}[split]
    path = Path(metadata_dir) / "sets" / f"{split_name}_mos_list.txt"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    annotations = {r[0]: (float(r[1]), float(r[2])) for r in rows}
    if len(annotations) != len(rows):
        raise ValueError("Original annotations contain duplicate filenames")
    if set(annotations) != {r["audio_filename"] for r in records}:
        raise ValueError("Dataset filenames do not match the original split")
    for row in records:
        if not np.allclose(annotations[row["audio_filename"]],
                           [row["mi_score"], row["ta_score"]], rtol=0, atol=1e-8):
            raise ValueError(f"Original MOS mismatch: {row['audio_filename']}")


def make_manifest(dataset, records, n_folds=5, seed=42):
    folds = stratified_splits(records, n_folds, seed)
    audio_files = []
    for row, record in zip(dataset, records):
        audio = row["audio"]
        blob = audio.get("bytes")
        if blob is None:
            path = Path(audio["path"])
            blob = path.read_bytes() if path.is_file() else None
        audio_files.append({"audio_filename": record["audio_filename"],
                            "sha256": hashlib.sha256(blob).hexdigest() if blob is not None else None})
    return {"protocol": PROTOCOL, "split": "train", "seed": seed,
            "manifest_format": 2, "audio_files": audio_files,
            "n_folds": n_folds, "n_samples": len(records),
            "n_systems": len({r["model_id"] for r in records}),
            "dataset_fingerprint": records_fingerprint(records),
            "records_sha256": records_fingerprint(records), "records": records,
            "folds": [{"fold": i, "train_indices": train, "val_indices": val,
                       "val_system_counts": dict(sorted(Counter(
                           records[j]["model_id"] for j in val).items()))}
                      for i, (train, val) in enumerate(folds)]}


def validate_manifest(manifest, dataset, records, n_folds, seed):
    expected = make_manifest(dataset, records, n_folds, seed)
    # Cache location is not identity. Ordered audio bytes, labels and folds are.
    if ({k: v for k, v in manifest.items() if k != "dataset_fingerprint"}
            != {k: v for k, v in expected.items() if k != "dataset_fingerprint"}):
        raise ValueError("Split manifest does not match dataset, seed, or true TTM protocol")
    return [(fold["train_indices"], fold["val_indices"]) for fold in manifest["folds"]]
