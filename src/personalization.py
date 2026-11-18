"""Author-initialized, last-four-attention LoRA preference adaptation."""

from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
from omegaconf import OmegaConf
from peft import LoraConfig
from scipy.stats import spearmanr
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .data_protocol import records_fingerprint
from .config import scientific_config
from .runtime import choose_gpu as select_gpu, bind_gpu
from .losses import GaussianOrdinalCELoss
from .model import MusicQualityModel
from .personalization_data import (
    PROTOCOL, PreferenceDataset, allowed_path, file_hash, save_csv, save_json, validate_manifest,
)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def identity_for(cfg, manifest):
    return {"protocol": manifest["protocol"], "manifest_sha256": records_fingerprint(manifest),
            "author_checkpoint_sha256": file_hash(cfg.personalization.author_checkpoint),
            "config_sha256": records_fingerprint(scientific_config(cfg))}


def choose_gpu(cfg):
    return select_gpu(cfg.personalization.preferred_gpu, cfg.personalization.minimum_free_gib)


def configure_gpu(cfg, stage):
    gpu, free = choose_gpu(cfg)
    bind_gpu(gpu, cfg.personalization.allocator_limit_gib)
    print(json.dumps({"stage": stage, "gpu": gpu, "free_mib": free,
                      "allocator_limit_gib": cfg.personalization.allocator_limit_gib,
                      "batch": cfg.training.batch_size, "precision": cfg.training.mixed_precision,
                      "output": str(cfg.paths.output_dir), "offline": True}), flush=True)
    return gpu


def load_author(cfg):
    model = MusicQualityModel(cfg)
    state = torch.load(allowed_path(cfg.personalization.author_checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    return model


def add_personal_adapter(model, cfg):
    encoder = model.encoder.encoder
    name = cfg.personalization.adapter_name
    encoder.add_adapter(name, LoraConfig(r=16, lora_alpha=32, lora_dropout=.1, bias="none",
                                        target_modules=cfg.personalization.target_pattern))
    encoder.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = ((f".{name}." in parameter_name and "lora_" in parameter_name)
                                   or parameter_name.startswith("encoder.pooling.")
                                   or parameter_name.startswith("heads.MI."))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    modules = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and name in m.lora_A]
    if count != cfg.personalization.expected_trainable_params or len(modules) != 16:
        raise ValueError(f"Incorrect adapter/trainable scope: {count} parameters, {len(modules)} modules")
    return {"trainable_parameters": count, "new_lora_parameters": 524288, "modules": modules,
            "active_adapter": name, "legacy_quantizer_adapter": "retained, inactive, frozen"}


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def frozen_hash(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name not in trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batchnorm_hash(model):
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key, value in module.state_dict().items():
                digest.update(f"{name}.{key}".encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preference_score(logits):
    if logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all():
        raise FloatingPointError("Invalid five-bin preference logits")
    probabilities = logits.float().softmax(-1)
    return (probabilities * torch.arange(1, 6, device=logits.device, dtype=torch.float32)).sum(-1)


def metrics(targets, scores):
    targets, scores = np.asarray(targets), np.asarray(scores)
    if not len(targets) or not np.isfinite(scores).all():
        raise FloatingPointError("Empty/non-finite preference predictions")
    errors = np.abs(scores - targets)
    per_class = {str(int(label)): {"n": int(np.sum(targets == label)),
                                 "mae": float(np.mean(errors[targets == label])),
                                 "mean_score": float(np.mean(scores[targets == label]))}
                 for label in sorted(set(targets))}
    result = {"n": len(targets), "mae": float(np.mean(errors)),
              "rmse": float(np.sqrt(np.mean((scores - targets) ** 2))),
              "macro_mae": float(np.mean([v["mae"] for v in per_class.values()])), "per_class": per_class}
    if len(set(targets)) > 1 and np.std(scores) > 0:
        result["srcc"] = float(spearmanr(targets, scores).statistic)
    return result


def loader_for(cfg, records, shuffle=False):
    return DataLoader(PreferenceDataset(cfg.paths.output_dir, records), batch_size=cfg.training.batch_size,
                      shuffle=shuffle, num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False,
                      generator=torch.Generator().manual_seed(int(cfg.seed)), persistent_workers=False)


@torch.no_grad()
def predict(model, loader, loss_fn=None):
    model.eval()
    scores, targets, identifiers, loss_sum = [], [], [], 0.
    for batch in loader:
        wave = batch["waveform"].cuda(non_blocking=True)
        target = batch["target"].float().cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(wave)["MI"]
            loss = loss_fn(logits.float(), target) if loss_fn is not None else None
        values = preference_score(logits).cpu().tolist()
        if not np.isfinite(values).all() or (loss is not None and not torch.isfinite(loss)):
            raise FloatingPointError("Non-finite preference validation/inference")
        scores.extend(values)
        targets.extend(batch["target"].tolist())
        identifiers.extend(batch["sample_id"])
        if loss is not None:
            loss_sum += float(loss) * len(values)
    expected = [r["sample_id"] for r in loader.dataset.records]
    if identifiers != expected:
        raise ValueError("Preference prediction order changed")
    return scores, metrics(targets, scores), loss_sum / len(scores) if loss_fn is not None else None


def gradient_audit(model):
    groups = {"lora": [], "pooling": [], "MI_head": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"Frozen parameter received a gradient: {name}")
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Missing/non-finite trainable gradient: {name}")
        group = "lora" if "lora_" in name else "pooling" if name.startswith("encoder.pooling.") else "MI_head"
        groups[group].append(parameter)
    if not all(groups.values()):
        raise RuntimeError("Missing a trainable parameter group")
    if not any(bool(p.grad.count_nonzero()) for p in groups["lora"]):
        raise RuntimeError("Active attention LoRA has only zero gradients")
    return {"frozen_parameters_have_gradients": False,
            **{f"{key}_gradient_tensors": len(values) for key, values in groups.items()}}


def save_torch(path, value):
    path = allowed_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_personal_checkpoint(model, path, identity):
    checkpoint = torch.load(allowed_path(path), map_location="cpu", weights_only=True)
    if checkpoint["identity"] != identity:
        raise ValueError("Checkpoint belongs to a different author/config/data protocol")
    if checkpoint["format"] == "personal_full_v1":
        model.load_state_dict(checkpoint["model_state"], strict=True)
    elif checkpoint["format"] == "personal_light_v1":
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(checkpoint["weights"]) != expected:
            raise ValueError("Lightweight checkpoint is missing trainable weights")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in expected:
                    parameter.copy_(checkpoint["weights"][name])
    else:
        raise ValueError("Unknown personalized checkpoint format")
    return checkpoint


def train_personal(cfg, manifest):
    root = allowed_path(cfg.paths.output_dir)
    if (root / "best_model.pt").exists() or (root / "training_results.json").exists():
        raise FileExistsError("Refusing to overwrite an existing/partial personal training run")
    validate_manifest(manifest, root, music_dir=cfg.personalization.music_dir)
    identity = identity_for(cfg, manifest)
    save_json(root / "run_identity.json", identity)
    gpu = configure_gpu(cfg, "train")
    seed_model(int(cfg.seed))
    started = time.perf_counter()
    status = {"status": "training", "started": now(), "gpu": gpu, "epochs_completed": 0}
    save_json(root / "run_status.json", status)
    try:
        # MuQ rotary embeddings cache tensors on first forward; move before any forward.
        model = load_author(cfg).cuda().eval()
        train_rows = [r for r in manifest["samples"] if r["split"] == "train"]
        val_rows = [r for r in manifest["samples"] if r["split"] == "validation"]
        train_loader, val_loader = loader_for(cfg, train_rows, True), loader_for(cfg, val_rows)
        author_scores, _, _ = predict(model, val_loader)
        scope = add_personal_adapter(model, cfg)
        init_scores, _, _ = predict(model, val_loader)
        initial_error = float(np.max(np.abs(np.asarray(author_scores) - init_scores)))
        if initial_error > 1e-6:
            raise ValueError(f"Zero-initialized personal adapter changed author predictions: {initial_error}")
        frozen_initial, bn_initial = frozen_hash(model), batchnorm_hash(model)
        groups = []
        for label, prefix, lr in (("lora", None, cfg.personalization.lora_lr),
                                  ("pooling", "encoder.pooling.", cfg.personalization.pooling_lr),
                                  ("MI_head", "heads.MI.", cfg.personalization.head_lr)):
            params = [p for n, p in model.named_parameters() if p.requires_grad
                      and (("lora_" in n) if prefix is None else n.startswith(prefix))]
            groups.append({"params": params, "lr": lr, "name": label})
        optimizer = AdamW(groups, weight_decay=cfg.training.weight_decay)
        if manifest["protocol"] == PROTOCOL and len(train_loader) != 8:
            raise ValueError("Expected exactly eight optimizer steps per epoch")
        total_steps = cfg.training.epochs * len(train_loader)
        scheduler = SequentialLR(optimizer, schedulers=[
            LinearLR(optimizer, start_factor=.1, total_iters=cfg.training.warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - cfg.training.warmup_steps, eta_min=1e-7)],
            milestones=[cfg.training.warmup_steps])
        loss_fn = GaussianOrdinalCELoss(num_bins=5, sigma=.5).cuda()
        history, best_key, stale, best_epoch, best_predictions, gradient_check = [], None, 0, 0, None, None
        torch.manual_seed(int(cfg.seed))
        torch.cuda.manual_seed_all(int(cfg.seed))
        for epoch in range(1, cfg.training.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            freeze_batchnorm(model)
            loss_sum, seen = 0., 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                target = batch["target"].float().cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch["waveform"].cuda(non_blocking=True))["MI"]
                    loss = loss_fn(logits.float(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
                loss.backward()
                check = gradient_audit(model)
                gradient_check = gradient_check or check
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                loss_sum += float(loss) * len(target)
                seen += len(target)
            scores, validation, val_loss = predict(model, val_loader, loss_fn)
            if batchnorm_hash(model) != bn_initial:
                raise RuntimeError("Frozen BatchNorm statistics changed")
            row = {"epoch": epoch, "train_loss": loss_sum / seen, "val_loss": val_loss,
                   "val_macro_mae": validation["macro_mae"], "val_mae": validation["mae"],
                   "val_rmse": validation["rmse"], "epoch_time_sec": time.perf_counter() - epoch_started,
                   **{f"lr_{g['name']}": float(g["lr"]) for g in optimizer.param_groups}}
            history.append(row)
            key = (validation["macro_mae"], validation["mae"])
            improved = best_key is None or key < best_key
            if improved:
                best_key, stale, best_epoch, best_predictions = key, 0, epoch, scores
                save_torch(root / "best_model.pt", {"format": "personal_full_v1", "identity": identity,
                    "model_state": model.state_dict(), "epoch": epoch, "validation_metrics": validation,
                    "validation_scores": scores, "scope": scope, "frozen_state_sha256": frozen_initial,
                    "config": OmegaConf.to_container(cfg, resolve=True)})
            else:
                stale += 1
            save_json(root / "training_history.json", history)
            save_csv(root / "training_history.csv", history)
            status.update(epochs_completed=epoch, best_epoch=best_epoch, latest_validation=validation)
            save_json(root / "run_status.json", status)
            print(json.dumps({**row, "best_epoch": best_epoch, "improved": improved}, ensure_ascii=False), flush=True)
            if stale >= cfg.training.patience:
                break
        optimizer.zero_grad(set_to_none=True)
        if frozen_hash(model) != frozen_initial:
            raise RuntimeError("A frozen parameter or buffer changed during personalization")
        checkpoint = load_personal_checkpoint(model, root / "best_model.pt", identity)
        reloaded_scores, validation, _ = predict(model, val_loader, loss_fn)
        error = float(np.max(np.abs(np.asarray(reloaded_scores) - best_predictions)))
        if error > 1e-4:
            raise RuntimeError(f"Best checkpoint reloading changed validation scores: {error}")
        weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        save_torch(root / "personal_weights.pt", {"format": "personal_light_v1", "identity": identity,
                    "weights": weights, "scope": scope, "epoch": best_epoch})
        results = {"identity": identity, "best_epoch": best_epoch, "epochs_run": len(history),
                   "optimizer_steps": len(history) * len(train_loader), "scheduled_maximum_steps": total_steps,
                   "warmup_steps": cfg.training.warmup_steps, "best_validation": validation,
                   "initialization_max_abs_error": initial_error, "reload_max_abs_error": error,
                   "gradient_check": gradient_check, "scope": scope, "frozen_state_sha256": frozen_initial,
                   "frozen_state_unchanged": True, "batchnorm_unchanged": True,
                   "demonstrations_used_for_selection": False, "gpu": gpu,
                   "wall_time_sec": time.perf_counter() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        save_json(root / "training_results.json", results)
        status.update(status="trained", finished_training=now())
        save_json(root / "run_status.json", status)
        del checkpoint, model, optimizer, scheduler, weights
        gc.collect()
        torch.cuda.empty_cache()
        return results
    except Exception as error:
        status.update(status="failed", error=repr(error))
        save_json(root / "run_status.json", status)
        raise


def evaluate_personal(cfg, manifest):
    root = allowed_path(cfg.paths.output_dir)
    validate_manifest(manifest, root, music_dir=cfg.personalization.music_dir)
    identity = identity_for(cfg, manifest)
    training = json.loads((root / "training_results.json").read_text())
    if identity != training["identity"]:
        raise ValueError("Training identity changed before evaluation")
    if not torch.cuda.is_initialized():
        configure_gpu(cfg, "evaluate")
    seed_model(int(cfg.seed))
    groups = {split: [r for r in manifest["samples"] if r["split"] == split]
              for split in ("train", "validation")}
    groups["demo"] = manifest["demonstrations"]
    model = load_author(cfg).cuda().eval()
    baseline = {split: predict(model, loader_for(cfg, rows))[0] for split, rows in groups.items()}
    add_personal_adapter(model, cfg)
    checkpoint = load_personal_checkpoint(model, root / "best_model.pt", identity)
    if frozen_hash(model) != checkpoint["frozen_state_sha256"]:
        raise RuntimeError("Reloaded personalized model differs in frozen state")
    after, statistics = {}, {}
    for split, records in groups.items():
        scores, metric, _ = predict(model, loader_for(cfg, records))
        after[split], statistics[split] = scores, metric
    error = float(np.max(np.abs(np.asarray(after["validation"]) - checkpoint["validation_scores"])))
    if error > 1e-4:
        raise RuntimeError(f"Freshly constructed model changed validation scores: {error}")
    repeated, _, _ = predict(model, loader_for(cfg, groups["demo"]))
    repeat_error = float(np.max(np.abs(np.asarray(repeated) - after["demo"])))
    if repeat_error > 1e-4:
        raise RuntimeError("Final WAV re-read scoring is not reproducible")
    # Loading the small export onto the exact author starting point must give the same result.
    del checkpoint, model
    gc.collect()
    torch.cuda.empty_cache()
    model = load_author(cfg).cuda().eval()
    add_personal_adapter(model, cfg)
    load_personal_checkpoint(model, root / "personal_weights.pt", identity)
    light_scores, _, _ = predict(model, loader_for(cfg, groups["demo"]))
    light_error = float(np.max(np.abs(np.asarray(light_scores) - after["demo"])))
    if light_error > 1e-4:
        raise RuntimeError("Lightweight export disagrees with the full checkpoint")
    rows = []
    for split, records in groups.items():
        for record, before, score in zip(records, baseline[split], after[split]):
            rows.append({**record, "author_MI_score": before, "preference_score": score,
                         "absolute_error": abs(score - record["preference_target"])})
    demos = [row for row in rows if row["split"] == "demo"]
    windows = [r["display_target_interval"] for r in manifest["demonstrations"]]
    for row, window in zip(demos, windows):
        row["display_target_interval"] = list(window)
        row["display_target_met"] = window[0] <= row["preference_score"] <= window[1]
    results = {"identity": identity, "selected_epoch": training["best_epoch"], "metrics": statistics,
               "author_baseline_metrics": {split: metrics([r["preference_target"] for r in records], baseline[split])
                                           for split, records in groups.items()},
               "fresh_reload_max_abs_error": error, "wav_reread_max_abs_error": repeat_error,
               "lightweight_max_abs_error": light_error, "demonstrations": demos,
               "display_targets_met": sum(r["display_target_met"] for r in demos),
               "display_targets_total": len(demos), "demonstrations_used_for_selection": False,
               "interpretation": "Same-song unseen-interval personal preference adaptation, not unseen-song generalization or objective music quality",
               "manual_audition": "not_performed"}
    save_json(root / "predictions.json", rows)
    save_csv(root / "predictions.csv", rows)
    save_csv(root / "demonstration_scores.csv", demos)
    save_json(root / "evaluation_results.json", results)
    status = json.loads((root / "run_status.json").read_text())
    status.update(status="evaluated", finished_evaluation=now(), display_targets_met=results["display_targets_met"])
    save_json(root / "run_status.json", status)
    print(json.dumps({"evaluation": statistics, "demonstrations": [
        {k: r[k] for k in ("title", "preference_target", "author_MI_score", "preference_score", "display_target_met")}
        for r in demos]}, ensure_ascii=False, indent=2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results
