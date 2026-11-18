"""Deterministic, fixed-strength audio interventions for the classroom study."""

import hashlib
import io

import numpy as np
from scipy import signal
import soundfile as sf


PROTOCOL = "a1_classroom_test_v1"
RATE = 24000
SAMPLES = 240000
SEED = 42
TOLERANCE = 1e-4
PEAK_LIMIT = 10 ** (-1 / 20)
METHODS = {
    "noise": {"label": "高斯噪声", "values": [20, 10, 0], "unit": "dB SNR"},
    "lowpass": {"label": "低通滤波", "values": [4000, 2000, 1000], "unit": "Hz"},
    "clipping": {"label": "削波失真", "values": [2.0, 1.0, 0.5], "unit": "× RMS"},
    "dropout": {"label": "随机断续", "values": [5, 15, 30], "unit": "段 × 100 ms"},
}


def seeded_rng(filename, stream):
    digest = hashlib.sha256(f"{SEED}:{filename}:{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def waveform_hash(waveform):
    return hashlib.sha256(np.asarray(waveform, dtype="<f4").tobytes()).hexdigest()


def pcm24_roundtrip(waveform):
    buffer = io.BytesIO()
    sf.write(buffer, waveform, RATE, format="WAV", subtype="PCM_24")
    buffer.seek(0)
    audio, rate = sf.read(buffer, dtype="float32")
    if rate != RATE or audio.shape != (SAMPLES,) or not np.isfinite(audio).all():
        raise ValueError("Invalid PCM-24 round trip")
    return audio


def shared_headroom(waveforms):
    peak = max(float(np.max(np.abs(w))) for w in waveforms)
    gain = min(1.0, PEAK_LIMIT / peak) if peak else 1.0
    return [np.asarray(w * gain, dtype=np.float32) for w in waveforms], gain


def make_variants(waveform, filename):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.shape != (SAMPLES,) or not np.isfinite(waveform).all():
        raise ValueError("Expected a finite 10-second mono waveform")
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    if rms < 1e-5:
        raise ValueError(f"Near-silent degradation source: {filename}")
    waves = [waveform.copy()]
    metadata = [{"variant": "clean", "method": "clean", "severity": 0,
                 "parameter": None, "diagnostics": {"source_rms": rms}}]
    noise = seeded_rng(filename, "noise").standard_normal(SAMPLES)
    noise /= np.sqrt(np.mean(noise ** 2))
    positions = seeded_rng(filename, "dropout").permutation(100)[:30].tolist()
    ramp = np.linspace(1, 0, int(0.005 * RATE), dtype=np.float32)
    for method, specification in METHODS.items():
        for severity, parameter in enumerate(specification["values"], 1):
            diagnostics = {}
            if method == "noise":
                addition = noise * rms * 10 ** (-parameter / 20)
                degraded = waveform.astype(np.float64) + addition
                diagnostics["measured_snr_db"] = float(10 * np.log10(
                    rms ** 2 / np.mean(addition ** 2)))
            elif method == "lowpass":
                sos = signal.butter(6, parameter, fs=RATE, btype="lowpass", output="sos")
                degraded = signal.sosfiltfilt(sos, waveform)
            elif method == "clipping":
                threshold = parameter * rms
                degraded = np.clip(waveform, -threshold, threshold)
                diagnostics = {"threshold": threshold,
                               "clipped_fraction": float(np.mean(np.abs(waveform) > threshold))}
            else:
                mask = np.ones(SAMPLES, dtype=np.float32)
                intervals = []
                for position in sorted(positions[:parameter]):
                    start, end = position * 2400, (position + 1) * 2400
                    mask[start:end] = 0
                    mask[start:start + len(ramp)] = ramp
                    mask[end - len(ramp):end] = ramp[::-1]
                    intervals.append([start, end])
                degraded = waveform * mask
                diagnostics = {"intervals_samples": intervals,
                               "zero_fraction": float(np.mean(mask == 0))}
            waves.append(np.asarray(degraded, dtype=np.float32))
            metadata.append({"variant": f"{method}_{severity}", "method": method,
                             "severity": severity, "parameter": parameter,
                             "diagnostics": diagnostics})
    waves, gain = shared_headroom(waves)
    # Every scored intervention takes the same PCM-24 path as the exported WAV.
    waves = [pcm24_roundtrip(w) for w in waves]
    for wave, row in zip(waves, metadata):
        row.update(common_gain=gain, common_gain_db=float(20 * np.log10(gain)),
                   waveform_sha256=waveform_hash(wave), peak=float(np.max(np.abs(wave))))
    return waves, metadata


def source_ranking(baselines, interventions):
    threshold = float(np.quantile([r["MI"] for r in baselines], .75))
    groups = {}
    for row in interventions:
        groups.setdefault(row["clip_id"], {})[row["variant"]] = row
    ranked = []
    for base in baselines:
        if base["clip_id"] not in groups or base["MI"] < threshold:
            continue
        rows = groups[base["clip_id"]]
        strict, weak, heavy = 0, 0, []
        for method in METHODS:
            values = [rows["clean"]["MI"]] + [rows[f"{method}_{s}"]["MI"] for s in (1, 2, 3)]
            strict += int(np.all(np.diff(values) < -TOLERANCE))
            weak += int(np.all(np.diff(values) <= TOLERANCE))
            heavy.append(values[0] - values[-1])
        ranked.append({**base, "strict_monotonic_methods": strict,
                       "nonincreasing_methods": weak, "mean_heavy_drop": float(np.mean(heavy))})
    ranked.sort(key=lambda r: (-r["strict_monotonic_methods"], -r["mean_heavy_drop"],
                               -r["MI"], r["audio_filename"]))
    selected, systems, prompts = [], set(), set()
    for row in ranked:
        if row["model_id"] not in systems and row["prompt_key"] not in prompts:
            selected.append(row["clip_id"])
            systems.add(row["model_id"])
            prompts.add(row["prompt_key"])
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError("Insufficient distinct high-quality sources; do not silently relax selection")
    return selected, ranked


def select_pairs(baselines):
    low, high = np.quantile([r["MI"] for r in baselines], [.25, .75])
    highs = [r for r in baselines if r["MI"] >= high and r["mi_score"] >= 3.8 and r["duration"] >= 10]
    lows = [r for r in baselines if r["MI"] <= low and r["mi_score"] <= 2.6 and r["duration"] >= 10]
    possibilities = [(h, l) for h in highs for l in lows]
    possibilities.sort(key=lambda pair: (pair[0]["prompt_key"] != pair[1]["prompt_key"],
                                         -(pair[0]["MI"] - pair[1]["MI"]),
                                         pair[0]["audio_filename"], pair[1]["audio_filename"]))
    selected, used, prompts = [], set(), set()
    for high_row, low_row in possibilities:
        if high_row["clip_id"] in used or low_row["clip_id"] in used:
            continue
        if high_row["prompt_key"] in prompts or low_row["prompt_key"] in prompts:
            continue
        selected.append({"high": high_row["clip_id"], "low": low_row["clip_id"],
                         "same_prompt": high_row["prompt_key"] == low_row["prompt_key"]})
        used.update([high_row["clip_id"], low_row["clip_id"]])
        prompts.update([high_row["prompt_key"], low_row["prompt_key"]])
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError("Insufficient genuine model/human high-low pairs")
    return selected


def summarize_interventions(rows):
    from scipy.stats import binomtest
    summary = {"sample_pool": "All official-test clips with original human MI >= Q75 and duration >=10s",
               "tie_tolerance": TOLERANCE, "methods": {}}
    for method, specification in METHODS.items():
        levels = {}
        for severity in (1, 2, 3):
            values = [r for r in rows if r["method"] == method and r["severity"] == severity]
            drops = np.asarray([r["drop_MI"] for r in values])
            count = int(np.sum(drops > TOLERANCE))
            ci = binomtest(count, len(values)).proportion_ci(method="wilson")
            levels[str(severity)] = {"n": len(values), "parameter": specification["values"][severity - 1],
                                     "mean_drop_MI": float(np.mean(drops)),
                                     "mean_drop_TA": float(np.mean([r["drop_TA"] for r in values])),
                                     "decrease_count": count, "decrease_rate": count / len(values),
                                     "wilson_95": [float(ci.low), float(ci.high)],
                                     "increase_count": int(np.sum(drops < -TOLERANCE)),
                                     "tie_count": int(np.sum(np.abs(drops) <= TOLERANCE))}
        per_source = {}
        for row in rows:
            if row["method"] in ("clean", method):
                per_source.setdefault(row["clip_id"], {})[row["severity"]] = row["MI"]
        changes = [np.diff([v[s] for s in range(4)]) for v in per_source.values()]
        summary["methods"][method] = {"label": specification["label"], "levels": levels,
                                     "strict_monotonic_rate": float(np.mean([np.all(x < -TOLERANCE) for x in changes])),
                                     "nonincreasing_rate": float(np.mean([np.all(x <= TOLERANCE) for x in changes]))}
    return summary
