import sys
sys.path.insert(0, '/home/pi/GrovePi/Software/Python')
sys.path.insert(0, '/home/pi/GrovePi/Software/Python/grove_rgb_lcd')
import subprocess
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
import csv
import time
import os
from collections import deque
import requests
from grove_rgb_lcd import *

# ── Config ────────────────────────────────────────────────────
SOUND_THRESHOLD = 0.02           # peak level to trigger classification
COOLDOWN        = 3              # seconds between alerts
MIN_SCORE       = 0.08           # minimum YAMNet confidence to print
BUFFER_CHUNKS   = 3              # rolling buffer size (3 x 1s = 3s of audio)
LAPTOP_IP = os.environ.get("LAPTOP_IP", "127.0.0.1")
WEBAPP_PORT = os.environ.get("WEBAPP_PORT", "5555")
WEBAPP_DETECTION_URL = os.environ.get(
    "WEBAPP_DETECTION_URL",
    f"http://{LAPTOP_IP}:{WEBAPP_PORT}/api/detection",
)

# Alert colors (R, G, B) for different sound types
ALERT_COLORS = {
    "siren":   (255, 0,   0),    # red
    "alarm":   (255, 0,   0),    # red
    "buzzer":  (255,0,0), 
    "knock":   (255, 165, 0),    # orange
    "speech":  (0,   0,   255),  # blue
    "default": (0,   200, 100),  # green
}

# ── Load YAMNet ──────────────────────────────────────────────
print("Loading YAMNet...")
interpreter = tflite.Interpreter(model_path="/home/pi/soniq/yamnet.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = []
with open("/home/pi/soniq/yamnet_class_map.csv", newline="") as f:
    for row in csv.DictReader(f):
        class_names.append(row["display_name"])

SKIP = {"Silence", "White noise", "Static", "Background noise", "Noise"}
print("YAMNet ready.")


# ── LCD setup ─────────────────────────────────────────────────
time.sleep(1)
setText("")
setRGB(0, 200, 100)  # green backlight at startup
print("LCD ready.\n")

# ── LCD helpers ───────────────────────────────────────────────
def lcd_16(s):
    s = str(s)
    return s[:16].ljust(16)

def show_on_lcd(label, score):
    top    = lcd_16("DETECTED:")
    bottom = lcd_16(f"{label} ({score:.2f})")
    setText_norefresh(top + "\n" + bottom)

    # Pick color based on label keyword
    color = ALERT_COLORS["default"]
    label_lower = label.lower()
    for key, rgb in ALERT_COLORS.items():
        if key in label_lower:
            color = rgb
            break
    setRGB(*color)

def clear_lcd():
    setText("")
    setRGB(0, 200, 100)

def send_detection_to_webapp(label, score):
    payload = {"sound": label, "score": float(score)}
    try:
        response = requests.post(WEBAPP_DETECTION_URL, json=payload, timeout=2)
        response.raise_for_status()
    except Exception as exc:
        # Keep detection loop resilient if network/webapp is unavailable.
        print(f" Failed to notify webapp: {exc}")

# ── Classify ─────────────────────────────────────────────────
def classify_sound(audio):
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
    top_indices = np.argsort(mean_scores)[::-1][:5]

    for idx in top_indices:
        label = class_names[idx]
        score = float(mean_scores[idx])
        if label not in SKIP and score > MIN_SCORE:
            return label, score

    return None, 0.0

# ── Record one chunk ──────────────────────────────────────────
def record_chunk():
    subprocess.run([
        "arecord", "-D", "plughw:1,0",
        "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", "1", "/tmp/chunk.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, _ = sf.read("/tmp/chunk.wav")
    return audio

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    last_alert = 0
    buffer = deque(maxlen=BUFFER_CHUNKS)

    print("Listening...\n")

    while True:
        # Always record into rolling buffer
        chunk = record_chunk()
        buffer.append(chunk)

        peak = float(np.max(np.abs(chunk)))
        print(f"Peak: {peak:.4f}", end="\r")

        if peak > SOUND_THRESHOLD and time.time() - last_alert > COOLDOWN:
            last_alert = time.time()
            print(f"\n Detected! (peak={peak:.3f})")

            # Classify buffered pre-trigger audio — captures the full sound onset
            buffered_audio = np.concatenate(list(buffer))
            label, score = classify_sound(buffered_audio)

            if label:
                print(f" {label}  ({score:.2f})\n")
                show_on_lcd(label, score)
                send_detection_to_webapp(label, score)
            else:
                print(" unknown\n")
                clear_lcd()
