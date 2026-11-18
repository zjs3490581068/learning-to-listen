"""Validate local assets and extract original annotations, without networking."""

import argparse
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import add_paths_argument, load_config
from src.data_protocol import check_original_annotations, dataset_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A1_true_ttm.yaml")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    from datasets import Audio, load_dataset
    archive_path = Path(cfg.storage.data_root) / "MusicEval/MusicEval-full.zip"
    metadata = Path(cfg.storage.data_root) / "MusicEval-original-metadata"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                relative = Path(item.filename)
                if item.is_dir() or not relative.parts or relative.parts[0] != "MusicEval-full" or "wav" in relative.parts:
                    continue
                if ".." in relative.parts or relative.is_absolute():
                    raise ValueError("Unsafe ZIP member")
                target = metadata.joinpath(*relative.parts[1:])
                if not target.resolve().is_relative_to(metadata.resolve()):
                    raise ValueError("ZIP member escapes metadata directory")
                content = archive.read(item)
                if target.exists():
                    if target.read_bytes() != content:
                        raise ValueError(f"Existing annotations differ: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
    for split, count, systems in (("train", 1923, 31), ("validation", 412, 31), ("test", 413, 30)):
        dataset = load_dataset(cfg.data.musiceval_id, split=split).cast_column("audio", Audio(decode=False))
        records = dataset_records(dataset)
        check_original_annotations(records, metadata, split)
        if len(records) != count or (split != "validation" and len({r['model_id'] for r in records}) != systems):
            raise ValueError(f"Unexpected official split: {split}")
        print(f"Verified {split}: {count} samples", flush=True)
    for name in ("config.json", "model.safetensors"):
        if not (Path(cfg.model.encoder_id) / name).is_file():
            raise FileNotFoundError(f"Missing base MuQ file: {name}")
    print("Local data and base model present. No weights loaded and no training performed.")


if __name__ == "__main__":
    main()
