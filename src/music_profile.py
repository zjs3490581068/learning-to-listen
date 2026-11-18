"""External personal-music protocol with no audio or storage locations."""

from copy import deepcopy
import hashlib
import json
from omegaconf import OmegaConf

from .config import project_path

ORIGINAL = "personalized_150_last4_v1"
CUSTOM = "personalized_custom_last4_v1"
ORIGINAL_PROFILE_SHA256 = "ee88a76efb7f11ade0d3d231112fddfeaa435874e3951120395f77419670a4b8"


def load_profile(path="configs/music.original.yaml"):
    profile = OmegaConf.to_container(OmegaConf.load(project_path(path)), resolve=True)
    validate_profile(profile)
    if profile["protocol"] == ORIGINAL:
        digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != ORIGINAL_PROFILE_SHA256:
            raise ValueError("Modified song protocols must use personalized_custom_last4_v1")
    return deepcopy(profile)


def validate_profile(profile):
    if profile["protocol"] not in (ORIGINAL, CUSTOM):
        raise ValueError("Unknown music profile protocol")
    count, val = profile["clips_per_song"], profile["validation_per_song"]
    if not isinstance(count, int) or not isinstance(val, int) or not 0 < val < count:
        raise ValueError("Require integer clip counts and nonempty train/validation")
    if (not all(type(profile[k]) is int for k in ("musiceval_validation", "musiceval_count"))
            or not 0 < profile["musiceval_validation"] < profile["musiceval_count"]):
        raise ValueError("Require nonempty MusicEval train and validation")
    titles = [s["title"] for s in profile["songs"]]
    if not titles or any(not isinstance(t, str) or not t.strip() for t in titles) or len(set(titles)) != len(titles) or "MusicEval" in titles:
        raise ValueError("Song titles must be nonempty and unique")
    for song in profile["songs"]:
        if type(song["preference_target"]) is not int or song["preference_target"] not in (1, 2, 3, 4, 5):
            raise ValueError("Preference labels must be integer bins 1..5")
        if not song.get("pattern") or song["pattern"].startswith("/") or ".." in song["pattern"].split("/"):
            raise ValueError("Song patterns must remain relative to music_root")
        protected = song.get("protected_interval")
        if protected is not None and (len(protected) != 2 or protected[0] < 0 or protected[1] - protected[0] != 10):
            raise ValueError("Protected demonstrations must be exactly ten seconds")
        window = song.get("display_target_interval") if protected else profile["musiceval_demo_interval"]
        if window is None or len(window) != 2 or not 1 <= window[0] <= window[1] <= 5:
            raise ValueError("Invalid demonstration target interval")
    expected = {s["title"] for s in profile["songs"] if s.get("protected_interval")} | {"MusicEval"}
    if len(profile["demo_order"]) != len(expected) or set(profile["demo_order"]) != expected:
        raise ValueError("demo_order must name each protected song and MusicEval exactly once")
    return True


def class_counts(profile):
    result = {"train": {}, "validation": {}}
    for song in profile["songs"]:
        label = song["preference_target"]
        for split, count in (("train", profile["clips_per_song"] - profile["validation_per_song"]),
                             ("validation", profile["validation_per_song"])):
            result[split][label] = result[split].get(label, 0) + count
    for split, count in (("train", profile["musiceval_count"] - profile["musiceval_validation"]),
                         ("validation", profile["musiceval_validation"])):
        result[split][1] = result[split].get(1, 0) + count
    return result
