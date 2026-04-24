import subprocess
import base64
import paho.mqtt.client as mqtt
import numpy as np
import soundfile as sf
import socket
import time

MQTT_TOPIC = "soniq/audio"


# ---------------------------
# CALLBACK
# ---------------------------
def on_connect(client, userdata, flags, rc):
    print("Connected to broker with result code", rc)

# ---------------------------
# AUDIO HELPERS
# ---------------------------
def record_chunk():
    subprocess.run([
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S16_LE",
        "-r", "16000",
        "-d", "1",   # 1 second chunk
        "chunk.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    audio, _ = sf.read("chunk.wav")
    return audio

def record_event():
    print("🎯 Recording event...")

    subprocess.run([
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S16_LE",
        "-r", "16000",
        "-d", "2",
        "event.wav"
    ])

    return "event.wav"

def compute_energy(audio):
    return np.mean(np.abs(audio)), np.max(np.abs(audio))

# ---------------------------
# MAIN
# ---------------------------
if __name__ == '__main__':

    client = mqtt.Client()
    client.on_connect = on_connect

    client.tls_set()
    client.connect("test.mosquitto.org", 8883, 60)

    client.loop_start()
    time.sleep(1)

    print("🎤 Listening...")

    while True:
        audio = record_chunk()

        energy, peak = compute_energy(audio)
        print(f"Energy: {energy:.4f}, Peak: {peak:.4f}", end="\r")

        # ✅ ONLY trigger when loud enough
        if peak > 0.05:
            print("\n🎯 Event detected!")

            wav_path = record_event()

            with open(wav_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            client.publish(MQTT_TOPIC, encoded)
            print("📡 Sent event\n")

        time.sleep(0.1)