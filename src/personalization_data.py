"""Fixed personal-preference clips, source provenance, and leakage checks."""

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from datasets import Audio, load_dataset
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from .data_protocol import check_original_annotations, dataset_records, records_fingerprint
from .music_profile import load_profile, validate_profile, class_counts, ORIGINAL

PROTOCOL = "personalized_150_last4_v1"
RATE, SAMPLES = 24000, 240000
PEAK_LIMIT = 10 ** (-1 / 20)


def allowed_path(path):
    return Path(path).expanduser().resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with allowed_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wave_hash(wave):
    return hashlib.sha256(np.asarray(wave, dtype="<f4").tobytes()).hexdigest()


def save_json(path, data):
    path = allowed_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with allowed_path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def rng(stream, seed=42):
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def choose_intervals(wave, protected=None, count=8):
    """Lexicographic interval DP: silence, anchor distance, then earlier starts."""
    if wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError("Expected finite mono audio")
    duration = len(wave) / RATE
    head, tail = (20, 10) if duration >= 110 else (10, 10)
    frames = len(wave) // 2400
    powers = np.mean(wave[:frames * 2400].reshape(frames, 2400).astype(np.float64) ** 2, axis=1)
    db = 10 * np.log10(np.maximum(powers, 1e-12))
    threshold = max(-60., float(np.quantile(db, .9) - 35))
    quiet = db < threshold
    candidates = []
    for start in range(head, math.floor(duration - tail - 10) + 1):
        if protected and overlaps((start, start + 10), protected):
            continue
        fraction = float(np.mean(quiet[start * 10:(start + 10) * 10]))
        rms_db = float(10 * np.log10(max(float(np.mean(powers[start * 10:(start + 10) * 10])), 1e-12)))
        if fraction <= .1 and rms_db >= -40:
            candidates.append({"start_sec": start, "end_sec": start + 10,
                               "quiet_fraction": fraction, "rms_dbfs": rms_db})
    anchors = np.linspace(head + 5, duration - tail - 5, count)
    previous = [None] * (len(candidates) + 1)
    previous[:] = [(0, 0., ())] * len(previous)
    starts = np.asarray([r["start_sec"] for r in candidates])
    ends = starts + 10
    for slot in range(count):
        current = [None] * (len(candidates) + 1)
        for j, row in enumerate(candidates):
            before = int(np.searchsorted(ends, row["start_sec"], side="right"))
            parent = previous[before]
            take = None
            if parent is not None:
                take = (parent[0] + round(row["quiet_fraction"] * 100),
                        parent[1] + abs(row["start_sec"] + 5 - anchors[slot]),
                        parent[2] + (j,))
            options = [v for v in (current[j], take) if v is not None]
            current[j + 1] = min(options) if options else None
        previous = current
    if not previous[-1]:
        raise ValueError(f"Cannot select {count} nonoverlapping eligible clips; candidates={len(candidates)}")
    selected = [candidates[i] for i in previous[-1][2]]
    return selected, {"head_excluded_sec": head, "tail_excluded_sec": tail,
                      "quiet_threshold_dbfs": threshold, "eligible_candidates": len(candidates),
                      "protected_interval": protected, "anchors_sec": anchors.tolist()}


def decode_source(source):
    array, rate = sf.read(source, dtype="float32", always_2d=True)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Empty/non-finite source audio")
    mono = torch.from_numpy(array.mean(axis=1))
    if rate != RATE:
        mono = torchaudio.functional.resample(mono, rate, RATE)
    wave = mono.numpy()
    peak = float(np.max(np.abs(wave)))
    if peak < 1e-5:
        raise ValueError("Near-silent source audio")
    gain = min(1., PEAK_LIMIT / peak)
    return wave, {"original_rate": rate, "original_channels": array.shape[1],
                  "original_frames": len(array), "duration_sec": len(array) / rate,
                  "resampled_samples": len(wave), "source_peak_24k": peak,
                  "common_gain": gain, "common_gain_db": float(20 * np.log10(gain))}


def export_clip(root, wave, row, start_sample):
    audio = np.asarray(wave[start_sample:start_sample + SAMPLES] * row["common_gain"], dtype=np.float32)
    if audio.shape != (SAMPLES,) or not np.isfinite(audio).all() or np.max(np.abs(audio)) > PEAK_LIMIT + 1e-6:
        raise ValueError("Invalid clip or unexpected overflow")
    relative = f"audio/{row['split']}/{row['sample_id']}.wav"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, RATE, subtype="PCM_24")
    actual, rate = sf.read(path, dtype="float32")
    if rate != RATE or actual.shape != (SAMPLES,) or np.max(np.abs(actual - audio)) > 2e-7:
        raise ValueError("PCM-24 export changed samples unexpectedly")
    return {**row, "path": relative, "start_sample_24k": start_sample,
            "end_sample_24k": start_sample + SAMPLES, "waveform_sha256": wave_hash(actual),
            "file_sha256": file_hash(path), "peak": float(np.max(np.abs(actual))),
            "clip_rms": float(np.sqrt(np.mean(actual.astype(np.float64) ** 2)))}


def validate_manifest(manifest, root=None, verify_sources=True, music_dir=None):
    profile = manifest.get("music_profile") or load_profile()
    validate_profile(profile)
    if profile["protocol"] == ORIGINAL and profile != load_profile():
        raise ValueError("Original music profile was modified")
    if manifest["protocol"] != profile["protocol"]:
        raise ValueError("Wrong personalization protocol")
    songs = {s["title"]: s for s in profile["songs"]}
    rows, demos = manifest["samples"], manifest["demonstrations"]
    count = len(songs) * profile["clips_per_song"] + profile["musiceval_count"]
    if len(rows) != count or len(demos) != len(profile["demo_order"]) or len({r["sample_id"] for r in rows + demos}) != count + len(demos):
        raise ValueError("Missing or duplicate samples")
    for split, counts in class_counts(profile).items():
        actual = Counter(r["preference_target"] for r in rows if r["split"] == split)
        if actual != counts:
            raise ValueError(f"Incorrect {split} class counts: {actual}")
    if [r["title"] for r in demos] != profile["demo_order"] or any(r["split"] != "demo" for r in demos):
        raise ValueError("Demonstration identities/order changed")
    if len({s["source_sha256"] for s in manifest["personal_sources"]}) != len(songs) or {s["title"] for s in manifest["personal_sources"]} != set(songs):
        raise ValueError("Missing or duplicate original personal songs")
    for source in manifest["personal_sources"]:
        title = source["title"]
        group = sorted((r for r in rows if r["source_id"] == source["source_id"]), key=lambda r: r["start_sec"])
        if len(group) != profile["clips_per_song"] or Counter(r["split"] for r in group) != {"train": profile["clips_per_song"] - profile["validation_per_song"], "validation": profile["validation_per_song"]}:
            raise ValueError("Incorrect per-song train/validation counts")
        for i, row in enumerate(group):
            if row["preference_target"] != songs[title]["preference_target"] or row["end_sec"] - row["start_sec"] != 10:
                raise ValueError("Personal label or duration changed")
            if row["start_sec"] < source["selection"]["head_excluded_sec"] or row["end_sec"] > source["duration_sec"] - source["selection"]["tail_excluded_sec"]:
                raise ValueError("Clip intrudes into excluded song edges")
            if row["quiet_fraction"] > .1 or row["rms_dbfs"] < -40:
                raise ValueError("Clip violates energy eligibility")
            if i and overlaps((group[i - 1]["start_sec"], group[i - 1]["end_sec"]), (row["start_sec"], row["end_sec"])):
                raise ValueError("Training/validation clips overlap")
            protected = songs[title].get("protected_interval")
            if protected and overlaps((row["start_sec"], row["end_sec"]), protected):
                raise ValueError("Demonstration audio leaked into training/validation")
            if row["common_gain"] != source["common_gain"]:
                raise ValueError("Inconsistent same-source gain")
        if verify_sources:
            if music_dir is None:
                raise ValueError("Source verification requires the current music_dir")
            path = allowed_path(Path(music_dir) / source["source_path"])
            if not path.is_relative_to(allowed_path(music_dir)) or file_hash(path) != source["source_sha256"]:
                raise ValueError("Original song was modified or escaped music_root")
    for demo in (d for d in demos if d["title"] != "MusicEval"):
        if [demo["start_sec"], demo["end_sec"]] != songs[demo["title"]]["protected_interval"]:
            raise ValueError("Fixed demonstration interval changed")
        if demo["preference_target"] != songs[demo["title"]]["preference_target"]:
            raise ValueError("Fixed demonstration target changed")
    music = [r for r in rows if r["source_kind"] == "musiceval"]
    if len(music) != profile["musiceval_count"] or len({r["audio_filename"] for r in music}) != len(music):
        raise ValueError("MusicEval training selection not unique/complete")
    if any(r["official_split"] != "train" or r["preference_target"] != 1 for r in music):
        raise ValueError("MusicEval source split or personal label changed")
    demo_me = next(d for d in demos if d["title"] == "MusicEval")
    if demo_me["official_split"] != "test" or demo_me["audio_filename"] in {r["audio_filename"] for r in music}:
        raise ValueError("MusicEval demo is not held out")
    if root:
        root = allowed_path(root)
        for row in rows + demos:
            path = allowed_path(root / row["path"])
            if not path.is_relative_to(root) or file_hash(path) != row["file_sha256"]:
                raise ValueError("Exported audio file/hash changed")
            wave, rate = sf.read(path, dtype="float32")
            if (rate != RATE or wave.shape != (SAMPLES,) or sf.info(path).subtype != "PCM_24"
                    or not np.isfinite(wave).all() or wave_hash(wave) != row["waveform_sha256"]
                    or np.max(np.abs(wave)) > PEAK_LIMIT + 1e-6):
                raise ValueError("Exported PCM-24 input failed verification")
    return True


def prepare_data(cfg):
    profile = load_profile(cfg.personalization.music_manifest)
    root = allowed_path(cfg.paths.output_dir)
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("music_profile") != profile:
            raise ValueError("Existing dataset belongs to a different music profile")
        validate_manifest(manifest, root, music_dir=cfg.personalization.music_dir)
        return manifest
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing/partial output: {root}")
    music_root = allowed_path(cfg.personalization.music_dir)
    identities = []
    for song in profile["songs"]:
        matches = sorted(music_root.glob(song["pattern"]))
        if len(matches) != 1 or matches[0].suffix.lower() != ".ogg" or not matches[0].resolve().is_relative_to(music_root):
            raise ValueError(f"Expected exactly one in-root Ogg for {song['title']}: {matches}")
        path = matches[0].resolve()
        if song.get("artist") and not path.stem.startswith(song["artist"] + " - "):
            raise ValueError(f"Unexpected artist/file identity: {path.name}")
        identities.append((path, song.get("artist", ""), song))
    identities.sort(key=lambda r: r[0])
    if len({r[0] for r in identities}) != len(identities):
        raise ValueError("Multiple songs matched the same recording")
    if profile["protocol"] == ORIGINAL and {r[0] for r in identities} != {p.resolve() for p in music_root.glob("*.ogg")}:
        raise ValueError("Original profile requires exactly the specified twelve Ogg files")
    root.mkdir(parents=True)
    samples, demos, sources = [], {}, []
    for number, (path, artist, song) in enumerate(identities):
        title = song["title"]
        wave, info = decode_source(path)
        score, protected = song["preference_target"], song.get("protected_interval")
        selected, selection = choose_intervals(wave, protected, profile["clips_per_song"])
        source_id = f"song_{number:02}"
        source = {"source_id": source_id, "source_kind": "personal", "title": title, "artist": artist,
                  "source_path": str(path.relative_to(music_root)), "audio_filename": path.name, "source_sha256": file_hash(path),
                  **info, "selection": selection}
        sources.append(source)
        validation_slots = set(rng(f"split:{path.name}", cfg.seed).permutation(profile["clips_per_song"])[-profile["validation_per_song"]:].tolist())
        shared = {k: v for k, v in source.items() if k != "selection"}
        for slot, interval in enumerate(selected):
            row = {**shared, **interval, "sample_id": f"{source_id}_{slot:02}",
                   "split": "validation" if slot in validation_slots else "train", "preference_target": score}
            samples.append(export_clip(root, wave, row, interval["start_sec"] * RATE))
        if protected:
            row = {**shared, "sample_id": f"demo_{source_id}", "split": "demo", "preference_target": score,
                   "start_sec": protected[0], "end_sec": protected[1],
                   "display_target_interval": song["display_target_interval"]}
            demos[title] = export_clip(root, wave, row, protected[0] * RATE)
        print(f"Prepared personal clips: {title} ({len(selected)} clips)", flush=True)
    fingerprints = {}
    for split, count in (("train", profile["musiceval_count"]), ("test", 1)):
        dataset = load_dataset(cfg.data.musiceval_id, split=split).cast_column("audio", Audio(decode=False))
        records = dataset_records(dataset)
        check_original_annotations(records, cfg.personalization.metadata_dir, split)
        fingerprints[split] = {"records_sha256": records_fingerprint(records)}
        eligible = []
        for record in sorted(records, key=lambda r: r["audio_filename"]):
            audio = dataset[record["clip_id"]]["audio"]
            buffer = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else allowed_path(audio["path"])
            if sf.info(buffer).duration >= 10:
                eligible.append(record)
        chosen = rng(f"musiceval:{split}", cfg.seed).choice(len(eligible), count, replace=False)
        for slot, position in enumerate(chosen):
            record = eligible[int(position)]
            audio = dataset[record["clip_id"]]["audio"]
            if audio.get("bytes") is not None:
                blob = audio["bytes"]
            else:
                blob = allowed_path(audio["path"]).read_bytes()
            wave, info = decode_source(io.BytesIO(blob))
            start = (len(wave) - SAMPLES) // 2
            row = {"sample_id": f"musiceval_{split}_{slot:02}", "source_id": record["audio_filename"],
                   "source_kind": "musiceval", "title": "MusicEval", "official_split": split,
                   "official_clip_id": record["clip_id"], "audio_filename": record["audio_filename"],
                   "model_id": record["model_id"], "original_human_MI": record["mi_score"],
                   "original_human_TA": record["ta_score"], "source_sha256": hashlib.sha256(blob).hexdigest(),
                   "preference_target": 1, "start_sec": start / RATE, "end_sec": (start + SAMPLES) / RATE,
                   "split": "demo" if split == "test" else "train" if slot < count - profile["musiceval_validation"] else "validation", **info}
            if split == "test":
                row["display_target_interval"] = profile["musiceval_demo_interval"]
            exported = export_clip(root, wave, row, start)
            if split == "test":
                demos["MusicEval"] = exported
            else:
                samples.append(exported)
    manifest = {"protocol": profile["protocol"], "music_profile": profile, "seed": int(cfg.seed), "personal_sources": sources, "samples": samples,
                "demonstrations": [demos[title] for title in profile["demo_order"]], "musiceval_provenance": fingerprints,
                "label_note": "Personal preference targets, not objective quality or original MusicEval MOS",
                "selection_note": "Energy/edge filtering only; no vocal detector and no manual audition",
                "sample_rate": RATE, "samples_per_clip": SAMPLES, "peak_limit": PEAK_LIMIT}
    validate_manifest(manifest, root, music_dir=music_root)
    save_json(manifest_path, manifest)
    save_csv(root / "clip_manifest.csv", samples + manifest["demonstrations"])
    print(f"Prepared and verified {class_counts(profile)}; {len(demos)} demonstrations", flush=True)
    return manifest


class PreferenceDataset(Dataset):
    def __init__(self, root, records):
        self.root = allowed_path(root)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        wave, rate = sf.read(self.root / record["path"], dtype="float32")
        if rate != RATE or wave.shape != (SAMPLES,) or not np.isfinite(wave).all():
            raise ValueError("Invalid prepared preference input")
        return {"waveform": torch.from_numpy(wave), "target": float(record["preference_target"]),
                "sample_id": record["sample_id"]}
