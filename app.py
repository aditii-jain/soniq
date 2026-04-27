from __future__ import annotations

import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NAME_WAV_PATH = PROJECT_ROOT / "name.wav"

# In-memory state so the UI can fetch latest selections and detections.
state = {
    "alerts": [],
    "last_detection": None,
    "updated_at": None,
}

ALLOWED_ALERT_TYPES = {"knock", "alarm", "name"}


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/alerts")
def set_alert_preferences():
    payload = request.get_json(silent=True) or {}
    selected = payload.get("alerts", [])
    if not isinstance(selected, list):
        return jsonify({"ok": False, "error": "alerts must be a list"}), 400

    normalized = []
    for label in selected:
        if not isinstance(label, str):
            continue
        label = label.strip().lower()
        if label in ALLOWED_ALERT_TYPES and label not in normalized:
            normalized.append(label)

    state["alerts"] = normalized
    state["updated_at"] = time.time()
    return jsonify({"ok": True, "alerts": state["alerts"]})


@app.post("/api/name-recording")
def save_name_recording():
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    raw_data = audio_file.read()
    if not raw_data:
        return jsonify({"ok": False, "error": "Empty audio file"}), 400

    if not raw_data.startswith(b"RIFF") or b"WAVE" not in raw_data[:32]:
        return jsonify({"ok": False, "error": "Uploaded audio must be WAV"}), 400

    NAME_WAV_PATH.write_bytes(raw_data)

    return jsonify({"ok": True, "path": str(NAME_WAV_PATH.name)})


@app.post("/api/detection")
def push_detection():
    payload = request.get_json(silent=True) or {}
    sound = payload.get("sound")
    score = payload.get("score")

    if not isinstance(sound, str) or not sound.strip():
        return jsonify({"ok": False, "error": "sound is required"}), 400

    state["last_detection"] = {
        "sound": sound.strip().lower(),
        "score": score,
        "at": time.time(),
    }
    state["updated_at"] = time.time()
    return jsonify({"ok": True, "last_detection": state["last_detection"]})


@app.get("/api/status")
def status():
    return jsonify(state)


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5555"))
    app.run(host=host, port=port, debug=True)
