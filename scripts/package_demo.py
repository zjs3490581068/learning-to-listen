"""Package a locally generated listening page, never training audio or weights."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import add_paths_argument, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("a1", "personal"), required=True)
    parser.add_argument("--config", help="Defaults to the selected experiment's configuration")
    parser.add_argument("--output-dir", help="Optional generated report directory")
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config or ("configs/A1_true_ttm.yaml" if args.kind == "a1" else "configs/A3a_personalized.yaml"), args.paths)
    root = Path(args.output_dir).resolve() if args.output_dir else (
        Path(cfg.storage.output_root) / "A1_classroom_degradation" if args.kind == "a1" else Path(cfg.paths.output_dir))
    common = {"index.html", "REPORT.md"}
    if args.kind == "a1":
        payload = json.loads((root / "classroom_cases.json").read_text())
        audio = [v["path"] for g in payload["sources"] + payload["limitations"] for v in g["variants"]]
        audio += [v["path"] for g in payload["pairs"] for v in g["entries"]]
        common.update(("baseline_predictions.csv", "degradation_predictions.csv", "selection.json"))
    else:
        payload = json.loads((root / "presentation.json").read_text())
        audio = [v["path"] for v in payload["demonstrations"]]
        common.update(("demonstration_scores.csv", "predictions.csv", "clip_manifest.csv"))
    common.update(audio)
    common.update(str(p.relative_to(root)) for p in (root / "figures").glob("*.png"))
    target = root / "classroom.zip"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {target}")
    paths = []
    for name in sorted(common):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe page resource: {name}")
        paths.append((path, name))
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for path, name in paths:
            archive.write(path, name)
    print(f"Packaged {len(paths)} presentation files: {target}; check audio redistribution permissions yourself")


if __name__ == "__main__":
    main()
