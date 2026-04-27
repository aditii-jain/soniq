import paho.mqtt.client as mqtt
import base64
import numpy as np
import soundfile as sf
import tensorflow_hub as hub
import tensorflow as tf
import os
import requests

model = hub.load("https://tfhub.dev/google/yamnet/1")
WEB_SERVER_DETECTION_URL = os.environ.get("WEB_SERVER_DETECTION_URL", "http://127.0.0.1:5000/api/detection")

def classify(audio, sr):
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    scores, _, _ = model(audio)
    scores = scores.numpy()
    mean_scores = scores.mean(axis=0)

    knock_score = np.max(mean_scores[350:460])
    alarm_score = np.max(mean_scores[300:350])

    return "knock" if knock_score > alarm_score else "alarm"

def on_message(client, userdata, msg):
    print("📥 Received audio")

    data = base64.b64decode(msg.payload)
    with open("recv.wav", "wb") as f:
        f.write(data)

    audio, sr = sf.read("recv.wav")
    pred = classify(audio, sr)

    print("🔍 Prediction:", pred)
    try:
        response = requests.post(
            WEB_SERVER_DETECTION_URL,
            json={"sound": pred},
            timeout=2,
        )
        response.raise_for_status()
    except Exception as exc:
        print("⚠️ Could not send detection to Flask server:", exc)

client = mqtt.Client()
client.on_message = on_message

client.connect("test.mosquitto.org", 1883)
client.subscribe("soniq/audio")

print("🟢 Listening for audio...")
client.loop_forever()