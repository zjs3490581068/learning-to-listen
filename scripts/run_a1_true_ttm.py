"""Run fold 0 first, then at most four independent, memory-bounded A1 folds."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets import Audio, load_dataset
from omegaconf import OmegaConf

from src.data_protocol import check_original_annotations, dataset_records, make_manifest, validate_manifest
from src.experiment_results import check_completed_fold, summarize_experiment
from train import aggregate_folds
from src.config import load_config, add_paths_argument
from src.config import scientific_config, project_path
from src.runtime import available_gpus as visible_gpus
from src.runtime import gpu_identifier


def available_gpus(excluded=(), requested=None):
    return visible_gpus(excluded, requested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/A1_true_ttm.yaml")
    parser.add_argument("--resume", action="store_true", help="Reuse completed folds; never overwrite partial folds")
    parser.add_argument("--gpus", help="Comma-separated physical GPU indices/UUIDs, within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max-parallel", type=int, choices=range(1, 5), default=1)
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    original = load_config("configs/A1_local.yaml", args.paths)
    if (scientific_config(cfg)["model"] != scientific_config(original)["model"] or cfg.training != original.training
            or cfg.loss != original.loss or cfg.seed != 42 or cfg.data.cv_folds != 5):
        raise ValueError("Refusing changed A1 architecture/hyperparameters or CV settings")
    dataset = load_dataset(cfg.data.musiceval_id, split="train").cast_column("audio", Audio(decode=False))
    records = dataset_records(dataset)
    check_original_annotations(records, cfg.data.musiceval_metadata_dir)
    manifest = make_manifest(dataset, records, cfg.data.cv_folds, cfg.seed)
    if manifest["n_samples"] != 1923 or manifest["n_systems"] != 31:
        raise ValueError("A1 requires the verified 1,923-clip / 31-system train split")
    if not all(item["sha256"] for item in manifest["audio_files"]):
        raise ValueError("A1 requires content checksums for every input audio file")
    output = Path(cfg.paths.output_dir) / cfg.experiment.name
    manifest_path = Path(cfg.data.split_manifest)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Existing output directory; use --resume only for completed folds: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        validate_manifest(json.loads(manifest_path.read_text()), dataset, records, 5, 42)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    logs = Path(cfg.storage.logs_root) / f"A1_true_ttm_{stamp}"
    logs.mkdir(parents=True, exist_ok=True)
    state = {"protocol": manifest["protocol"], "logs": str(logs), "output": str(output),
             "status": "running", "completed": [], "running": {}, "failed": []}
    active = {}

    def save_status():
        state["running"] = {str(fold): {"pid": item["process"].pid, "gpu": item["gpu"],
                                       "log": str(item["log"])} for fold, item in active.items()}
        (output / "run_status.json").write_text(json.dumps(state, indent=2) + "\n")

    def launch(fold, gpu):
        directory = output / f"fold{fold}"
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Partial/existing fold will not be overwritten: {directory}")
        command = [sys.executable, "-u", "train.py", "--config", str(project_path(args.config)),
                   "--fold", str(fold), "--max-gpu-memory-gib", "6"]
        if args.paths:
            command.extend(["--paths", str(project_path(args.paths))])
        log_path = logs / f"fold{fold}_gpu{gpu}.log"
        stream = log_path.open("w")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_identifier(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        print(json.dumps({"launch_fold": fold, "gpu": gpu, "command": command,
                          "cwd": str(REPO), "environment": sys.prefix,
                          "output": str(directory), "log": str(log_path)}, ensure_ascii=True), flush=True)
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT)
        active[fold] = {"process": process, "gpu": gpu, "stream": stream, "log": log_path}
        save_status()

    def reap():
        for fold, item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["stream"].close()
            del active[fold]
            try:
                if code:
                    raise RuntimeError(f"Worker exit code {code}; see {item['log']}")
                check_completed_fold(output / f"fold{fold}", fold, manifest)
                state["completed"].append(fold)
                print(f"Fold {fold} completed and passed all gates", flush=True)
            except Exception as error:
                state["failed"].append({"fold": fold, "error": str(error)})
                print(f"Fold {fold} FAILED: {error}", flush=True)
            save_status()

    if args.resume:
        for fold in range(5):
            if (output / f"fold{fold}").exists():
                check_completed_fold(output / f"fold{fold}", fold, manifest)
                state["completed"].append(fold)
    save_status()
    print(f"Verified manifest: {manifest_path}; worker logs: {logs}", flush=True)
    if 0 not in state["completed"]:
        if not visible_gpus(requested=args.gpus, minimum_free_gib=0):
            raise RuntimeError("No GPU matches the requested/visible device set")
        while not available_gpus(requested=args.gpus):
            print("Waiting for a GPU with >=8 GiB free", flush=True)
            time.sleep(30)
        launch(0, available_gpus(requested=args.gpus)[0][0])
        while active:
            time.sleep(5)
            reap()
        if state["failed"]:
            state["status"] = "failed_fold0_gate"
            save_status()
            return 1
    pending = [fold for fold in range(1, 5) if fold not in state["completed"]]
    while pending or active:
        reap()
        free = available_gpus({item["gpu"] for item in active.values()}, args.gpus)
        for gpu, _ in free:
            if not pending or len(active) >= args.max_parallel:
                break
            launch(pending.pop(0), gpu)
        if pending and not active:
            print("Waiting for a GPU with >=8 GiB free", flush=True)
        if pending or active:
            time.sleep(5 if active else 30)
    if state["failed"]:
        state["status"] = "failed"
        save_status()
        return 1
    aggregate_folds(cfg, 5)
    summary = summarize_experiment(output, manifest)
    state["status"] = "complete"
    save_status()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
