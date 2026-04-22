import librosa
import numpy as np
import sounddevice as sd

# ---------------------------
# CONFIG
# ---------------------------
SAMPLE_RATE = 22050
EVENT_START_DB = -35.0
EVENT_END_DB = -40.0
SILENCE_HOLD_SEC = 0.35
MIN_SIMILARITY = 0.55
MIN_SIM_MARGIN = 0.08
READ_CHUNK_SEC = 0.25
MIN_EVENT_SEC = 0.12
MAX_EVENT_SEC = 3.0

TEMPLATE_FILES = {
    "knock": "audio_templates/knock.wav",
    "alarm": "audio_templates/alarm.wav",
    # "speech": "audio_templates/speech.wav"
}

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def preprocess_audio(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if y_trimmed.size == 0:
        return y_trimmed

    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_trimmed = y_trimmed / peak
    return y_trimmed


def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    feat = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(rolloff), np.std(rolloff)],
        [np.mean(zcr), np.std(zcr)],
    ])
    return feat


# ---------------------------
# LOAD TEMPLATES
# ---------------------------
def load_templates():
    templates = {}
    for label, file in TEMPLATE_FILES.items():
        try:
            y, sr = librosa.load(file, sr=SAMPLE_RATE)
            y = preprocess_audio(y)
            if y.size == 0:
                print(f"Warning: template '{label}' in {file} is mostly silence; skipping.")
                continue
            templates[label] = extract_features(y, sr)
        except Exception as exc:
            print(f"Warning: could not load template '{label}' from {file}: {exc}")
    if not templates:
        raise ValueError("No valid audio templates loaded. Check TEMPLATE_FILES and audio files.")
    print(f"Loaded templates: {list(templates.keys())}")
    return templates

def compute_rms_db(y):
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)
# ---------------------------
# CLASSIFICATION
# ---------------------------
def classify(input_feat, templates):
    scores = {}
    for label, temp_feat in templates.items():
        sim = np.dot(input_feat, temp_feat) / (
            np.linalg.norm(input_feat) * np.linalg.norm(temp_feat) + 1e-12
        )
        scores[label] = float(sim)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else -1.0

    if best_score < MIN_SIMILARITY or (best_score - second_score) < MIN_SIM_MARGIN:
        return "unknown", scores
    return best_label, scores


# ---------------------------
# MAIN LOOP
# ---------------------------
def main():
    templates = load_templates()
    chunk_samples = int(READ_CHUNK_SEC * SAMPLE_RATE)
    silence_hold_chunks = max(1, int(round(SILENCE_HOLD_SEC / READ_CHUNK_SEC)))

    in_event = False
    event_chunks = []
    quiet_chunks_in_event = 0

    print("🎤 Continuous listening... Press Ctrl+C to stop.")

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

                y = preprocess_audio(event_audio)
                event_chunks = []
                if y.size == 0:
                    continue

                feat = extract_features(y, SAMPLE_RATE)
                pred, scores = classify(feat, templates)

                if pred == "unknown":
                    continue

                print(f"\n🔍 Prediction: {pred}")
                print(f"Similarity scores: {scores}")
    except KeyboardInterrupt:
        print("\nStopped listening.")


if __name__ == "__main__":
    main()