from __future__ import annotations

import csv
import io
import os
import sys
import urllib.request

import librosa
import numpy as np
import sounddevice as sd

# TensorFlow is loaded lazily in load_yamnet() with a clear error if missing.
tf = None
hub = None

# ---------------------------
# CONFIG
# ---------------------------
SAMPLE_RATE = 22050
YAMNET_SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

EVENT_START_DB = -35.0
EVENT_END_DB = -40.0
SILENCE_HOLD_SEC = 0.35
READ_CHUNK_SEC = 0.25
MIN_EVENT_SEC = 0.12
MAX_EVENT_SEC = 3.0

# YAMNet: multi-label sigmoid per class. Tune on your mic + room.
MIN_GROUP_SCORE = 0.4
MIN_GROUP_MARGIN = 0.04

# Optional: path to a local yamnet_class_map.csv (same as TensorFlow models repo)
YAMNET_CLASS_MAP_PATH = os.environ.get("YAMNET_CLASS_MAP", "")

# Fallback if class map cannot be loaded (AudioSet / YAMNet index order is stable for hub v1)
KNOCK_CLASS_INDICES_FALLBACK = {353, 354, 454, 455, 460}
ALARM_CLASS_INDICES_FALLBACK = {304, 312, 313, 349, 350, 382, 389, 390, 391, 392, 393, 394, 395, 475}

# ---------------------------
# YAMNet class grouping (from display_name in yamnet_class_map.csv)
# ---------------------------
def _is_knock_class(name: str) -> bool:
    n = (name or "").lower()
    if "engine knocking" in n or "backfire" in n:
        return False
    if n == "knock":
        return True
    if "thump" in n or "thud" in n or "thunk" in n:
        return True
    if n == "tap" or n.startswith("tap "):
        return True
    if n == "bang" or n.startswith("bang "):
        return True
    if "whack" in n or "thwack" in n or "slap" in n or "smack" in n:
        return True
    if n == "slam" or n.startswith("slam "):
        return True
    return False


def _is_alarm_class(name: str) -> bool:
    n = (name or "").lower()
    if "alarm" in n:
        return True
    if "siren" in n:
        return True
    if "buzzer" in n:
        return True
    if "smoke detector" in n or "smoke alarm" in n or "fire alarm" in n:
        return True
    if "foghorn" in n or "civil defense" in n:
        return True
    if "reversing beep" in n or "reversing beeps" in n:
        return True
    if "air horn" in n or "truck horn" in n or "vehicle horn" in n or "car horn" in n:
        return True
    if n == "beep" or "bleep" in n or "beep" in n:
        return True
    if n == "doorbell" or "ding-dong" in n or "ding dong" in n:
        return True
    return False


def _read_class_map_rows(text: str):
    rows = []
    f = io.StringIO(text)
    r = csv.DictReader(f, skipinitialspace=True)
    for row in r:
        idx = int(row["index"].strip())
        name = (row.get("display_name") or row.get("display name") or "").strip()
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        rows.append((idx, name))
    return rows


def _load_yamnet_class_map_text() -> str:
    if YAMNET_CLASS_MAP_PATH and os.path.isfile(YAMNET_CLASS_MAP_PATH):
        with open(YAMNET_CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            return f.read()
    req = urllib.request.Request(
        YAMNET_CLASS_MAP_URL,
        headers={"User-Agent": "soundsense-yamnet/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def build_yamnet_group_indices() -> tuple[set[int], set[int], list[str]]:
    knock: set[int] = set()
    alarm: set[int] = set()
    names: list[str] = [""] * 521
    try:
        text = _load_yamnet_class_map_text()
        for idx, name in _read_class_map_rows(text):
            if 0 <= idx < 521:
                names[idx] = name
            if _is_knock_class(name):
                knock.add(idx)
            if _is_alarm_class(name):
                alarm.add(idx)
    except Exception as exc:
        print(f"Warning: could not load YAMNet class map ({exc}). Using fallback class index sets.")
        return set(KNOCK_CLASS_INDICES_FALLBACK), set(ALARM_CLASS_INDICES_FALLBACK), names

    if not knock:
        knock = set(KNOCK_CLASS_INDICES_FALLBACK)
    if not alarm:
        alarm = set(ALARM_CLASS_INDICES_FALLBACK)
    return knock, alarm, names


def load_yamnet():
    global tf, hub
    try:
        import tensorflow as tf_mod
        import tensorflow_hub as hub_mod
    except Exception as exc:
        print("Could not import TensorFlow. Install with:")
        print("  pip install -r requirements.txt")
        print(f"({exc})")
        sys.exit(1)
    tf = tf_mod
    hub = hub_mod
    model = hub.load(YAMNET_HANDLE)
    return model


# ---------------------------
# Preprocess + YAMNet inference
# ---------------------------
def preprocess_for_yamnet(y: np.ndarray, sr: int) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if y_trimmed.size == 0:
        return y_trimmed.astype(np.float32)

    y_16k = librosa.resample(
        y_trimmed.astype(np.float32),
        orig_sr=sr,
        target_sr=YAMNET_SR,
    )
    peak = float(np.max(np.abs(y_16k)))
    if peak > 0.0:
        y_16k = (y_16k / peak * 0.98).astype(np.float32)
    else:
        y_16k = y_16k.astype(np.float32)

    min_samples = int(0.2 * YAMNET_SR)
    if y_16k.size < min_samples:
        y_16k = np.pad(y_16k, (0, min_samples - y_16k.size), mode="constant")
    return y_16k


def yamnet_class_scores(wave_16k: np.ndarray, model) -> np.ndarray:
    w = tf.convert_to_tensor(wave_16k, dtype=tf.float32)
    scores, _, _ = model(w)
    s = scores.numpy()
    if s.ndim == 1:
        return s
    return np.max(s, axis=0)


def yamnet_group_scores(
    per_class: np.ndarray,
    knock_idx: set[int],
    alarm_idx: set[int],
) -> dict[str, float]:
    if per_class.size < 521:
        per_class = np.pad(per_class, (0, 521 - per_class.size), mode="constant")
    k = float(np.max(per_class[list(knock_idx)])) if knock_idx else 0.0
    a = float(np.max(per_class[list(alarm_idx)])) if alarm_idx else 0.0
    return {"knock": k, "alarm": a}


def yamnet_classify(
    per_class: np.ndarray,
    knock_idx: set[int],
    alarm_idx: set[int],
) -> tuple[str, dict[str, float]]:
    scores = yamnet_group_scores(per_class, knock_idx, alarm_idx)
    k = scores["knock"]
    a = scores["alarm"]
    if max(k, a) < MIN_GROUP_SCORE:
        return "unknown", scores
    if abs(k - a) < MIN_GROUP_MARGIN:
        return "unknown", scores
    return ("knock" if k > a else "alarm"), scores


def compute_rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


# ---------------------------
# MAIN LOOP
# ---------------------------
def main():
    knock_idx, alarm_idx, _names = build_yamnet_group_indices()
    model = load_yamnet()
    print(
        f"YAMNet: {len(knock_idx)} knock-related classes, {len(alarm_idx)} alarm-related classes (AudioSet groupings)."
    )
    print(f"Config: min_score={MIN_GROUP_SCORE}, margin={MIN_GROUP_MARGIN}")
    print("🎤 Continuous listening (YAMNet). Press Ctrl+C to stop.\n")

    chunk_samples = int(READ_CHUNK_SEC * SAMPLE_RATE)
    silence_hold_chunks = max(1, int(round(SILENCE_HOLD_SEC / READ_CHUNK_SEC)))

    in_event = False
    event_chunks: list[np.ndarray] = []
    quiet_chunks_in_event = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:
            while True:
                chunk, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("Warning: audio buffer overflow detected.")

                chunk = np.asarray(chunk).flatten()
                chunk_db = compute_rms_db(chunk)

                if not in_event:
                    if chunk_db >= EVENT_START_DB:
                        in_event = True
                        event_chunks = [chunk]
                        quiet_chunks_in_event = 0
                    continue

                event_chunks.append(chunk)
                event_audio = np.concatenate(event_chunks)
                event_duration = len(event_audio) / SAMPLE_RATE

                if chunk_db < EVENT_END_DB:
                    quiet_chunks_in_event += 1
                else:
                    quiet_chunks_in_event = 0

                event_finished = quiet_chunks_in_event >= silence_hold_chunks
                event_too_long = event_duration >= MAX_EVENT_SEC
                if not event_finished and not event_too_long:
                    continue

                in_event = False
                quiet_chunks_in_event = 0

                if event_duration < MIN_EVENT_SEC:
                    event_chunks = []
                    continue

                e_audio = event_audio
                event_chunks = []
                e_audio = preprocess_for_yamnet(e_audio, SAMPLE_RATE)
                if e_audio.size == 0:
                    continue

                per_class = yamnet_class_scores(e_audio, model)
                pred, scores = yamnet_classify(per_class, knock_idx, alarm_idx)
                if pred == "unknown":
                    continue
                print(f"🔍 Prediction: {pred}")
                print(f"YAMNet group scores: {scores}\n")
    except KeyboardInterrupt:
        print("\nStopped listening.")


if __name__ == "__main__":
    main()
