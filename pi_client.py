import subprocess
import base64
import paho.mqtt.client as mqtt
import numpy as np
import soundfile as sf

MQTT_TOPIC = "soniq/audio"
client = mqtt.Client()
client.connect("test.mosquitto.org", 1883)

def record_chunk():
    subprocess.run([
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S16_LE",
        "-r", "16000",
        "-d", "0.5",
        "chunk.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    audio, _ = sf.read("chunk.wav")
    return audio

def compute_energy(audio):
    return np.mean(np.abs(audio)), np.max(np.abs(audio))

print("🎤 Listening...")

while True:
    audio = record_chunk()

    energy, peak = compute_energy(audio)
    print(f"Energy: {energy:.4f}, Peak: {peak:.4f}", end="\r")

    if peak > 0.05:
        print("\n🎯 Event detected!")

        subprocess.run([
            "arecord",
            "-D", "plughw:1,0",
            "-f", "S16_LE",
            "-r", "16000",
            "-d", "2",
            "event.wav"
        ])

        with open("event.wav", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        client.publish(MQTT_TOPIC, encoded)
        print("📡 Sent event")