import paho.mqtt.client as mqtt
import base64
import numpy as np
import soundfile as sf
import tensorflow_hub as hub

print("Loading model...")
model = hub.load("https://tfhub.dev/google/yamnet/1")
print("Model loaded.\n")

MQTT_TOPIC = "soniq/audio"
BROKER = "localhost"

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

def on_connect(client, userdata, flags, rc):
    print("Connected to broker:", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    print("\n📥 Received audio")

    try:
        data = base64.b64decode(msg.payload)

        with open("recv.wav", "wb") as f:
            f.write(data)

        audio, sr = sf.read("recv.wav")

        pred = classify(audio, sr)

        print("🔍 Prediction:", pred)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, 1883, 60)

    print("🟢 Listening...\n")
    client.loop_forever()