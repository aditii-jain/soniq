import subprocess
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
import csv
import time

# ── Load model ───────────────────────────────────────────────
print("Loading YAMNet...")
interpreter = tflite.Interpreter(model_path="/home/pi/soniq/yamnet.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = []
with open("/home/pi/soniq/yamnet_class_map.csv", newline="") as f:
    for row in csv.DictReader(f):
        class_names.append(row["display_name"])

print(f"✅ Model ready. Listening...\n")

SKIP_CLASSES = {"Silence", "White noise", "Static", "Background noise"}

# ── Classification ───────────────────────────────────────────
def classify(audio):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    chunk_size = 15600
    all_scores = []

    for start in range(0, len(audio) - chunk_size + 1, chunk_size):
        chunk = audio[start:start + chunk_size]
        interpreter.set_tensor(input_details[0]['index'], chunk)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]['index'])
        all_scores.append(scores[0])

    if not all_scores:
        return None, 0.0

    mean_scores = np.mean(all_scores, axis=0)

    # Get top 3 predictions
    top_indices = np.argsort(mean_scores)[::-1][:3]
    for idx in top_indices:
        label = class_names[idx]
        score = float(mean_scores[idx])
        if label not in SKIP_CLASSES and score > 0.1:
            return label, score

    return None, 0.0

# ── Audio ────────────────────────────────────────────────────
def record_chunk():
    subprocess.run([
        "arecord", "-D", "plughw:1,0",
        "-f", "S16_LE", "-r", "16000",
        "-d", "1", "/tmp/chunk.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, _ = sf.read("/tmp/chunk.wav")
    return audio

def record_event():
    subprocess.run([
        "arecord", "-D", "plughw:1,0",
        "-f", "S16_LE", "-r", "16000",
        "-d", "2", "/tmp/event.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sf.read("/tmp/event.wav")

# ── Main loop ────────────────────────────────────────────────
if __name__ == "__main__":
    last_event_time = 0

    while True:
        audio = record_chunk()
        peak  = float(np.max(np.abs(audio)))

        if peak > 0.05 and time.time() - last_event_time > 3:
            last_event_time = time.time()

            audio, sr = record_event()
            label, score = classify(audio)

            if label:
                print(f"🔍 {label}  ({score:.2f})")

        time.sleep(0.1)
