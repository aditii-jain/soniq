import paho.mqtt.client as mqtt
import base64
import numpy as np
import soundfile as sf
import tensorflow_hub as hub
import tensorflow as tf

# ---------------------------
# LOAD MODEL (once)
# ---------------------------
print("Loading YAMNet...")
model = hub.load("https://tfhub.dev/google/yamnet/1")
print("Model loaded.\n")

# ---------------------------
# CLASSIFICATION
# ---------------------------
def classify(audio, sr):
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    scores, _, _ = model(audio)
    scores = scores.numpy()
    mean_scores = scores.mean(axis=0)

    knock_score = np.max(mean_scores[350:460])
    alarm_score = np.max(mean_scores[300:350])

    if max(knock_score, alarm_score) < 0.3:
        return "unknown"

    return "knock" if knock_score > alarm_score else "alarm"

# ---------------------------
# CALLBACKS (EE250 STYLE)
# ---------------------------

def on_connect(client, userdata, flags, rc):
    print("Connected to broker with result code:", rc)

    # subscribe to your topic
    client.subscribe("soniq/audio")

    # attach custom callback
    client.message_callback_add("soniq/audio", on_message_audio)

def on_message(client, userdata, msg):
    print("Default callback:", msg.topic)

# custom callback
def on_message_audio(client, userdata, msg):
    print("📥 Received audio")

    try:
        data = base64.b64decode(msg.payload)

        with open("recv.wav", "wb") as f:
            f.write(data)

        audio, sr = sf.read("recv.wav")

        pred = classify(audio, sr)

        print("🔍 Prediction:", pred, "\n")

    except Exception as e:
        print("Error processing audio:", e)

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    # 🔥 CONNECT TO PI BROKER
    client.connect(host="test.mosquitto.org", port=1883, keepalive=60)

    print("🟢 Listening for audio...\n")

    client.loop_forever()