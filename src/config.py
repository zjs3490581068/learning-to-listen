"""Portable experiment configuration; importing this module has no side effects."""

import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"Circular configuration inheritance: {path}")
    cfg = OmegaConf.load(path)
    merged = OmegaConf.create()
    for name in cfg.pop("defaults", []):
        merged = OmegaConf.merge(merged, _read(path.parent / f"{name}.yaml", (*ancestors, path)))
    return OmegaConf.merge(merged, cfg)


def add_paths_argument(parser):
    parser.add_argument("--paths", help="Storage-only YAML override; relative paths use the project root")


def load_config(config_path, paths_path=None):
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs/paths.example.yaml"), _read(project_path(config_path)))
    if paths_path:
        override = OmegaConf.load(project_path(paths_path))
        if set(override) != {"storage"} or set(override.storage) - set(cfg.storage):
            raise ValueError("--paths accepts only the documented storage keys")
        cfg = OmegaConf.merge(cfg, override)
    for key in cfg.storage:
        cfg.storage[key] = str(project_path(cfg.storage[key]))
    OmegaConf.resolve(cfg)
    configure_environment(cfg)
    return cfg


def configure_environment(cfg):
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1")
    cache = Path(cfg.storage.cache_root)
    os.environ.update(HF_HOME=str(cache / "huggingface"), HF_HUB_CACHE=str(cache / "huggingface/hub"),
                      HF_DATASETS_CACHE=str(cache / "huggingface/datasets"), TORCH_HOME=str(cache / "torch"))
    # Hub/datasets may already have been imported by an entry point.
    for name, attribute in (("huggingface_hub.constants", "HF_HUB_OFFLINE"),
                            ("datasets.config", "HF_DATASETS_OFFLINE")):
        if name in sys.modules:
            setattr(sys.modules[name], attribute, True)
    if "datasets.config" in sys.modules:
        sys.modules["datasets.config"].HF_DATASETS_CACHE = Path(os.environ["HF_DATASETS_CACHE"])
    if "huggingface_hub.constants" in sys.modules:
        hub = sys.modules["huggingface_hub.constants"]
        hub.HF_HOME = os.environ["HF_HOME"]
        hub.HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]
        hub.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
        hub.HF_HUB_DISABLE_TELEMETRY = True


def scientific_config(cfg):
    """Storage relocation must not change checkpoint identity."""
    value = OmegaConf.to_container(cfg, resolve=True)
    value.pop("storage", None)
    value.pop("paths", None)
    for section, keys in {
        "data": ("musiceval_id", "musiceval_metadata_dir", "split_manifest"),
        "model": ("encoder_id",),
        "personalization": ("music_dir", "author_checkpoint", "metadata_dir", "music_manifest", "preferred_gpu"),
    }.items():
        for key in keys:
            value.get(section, {}).pop(key, None)
    return value
