import sounddevice as sd
import soundfile as sf
import paho.mqtt.client as mqtt
import base64
import numpy as np

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
THRESHOLD = 0.02

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883)

def compute_energy(chunk):
    return np.mean(np.abs(chunk))

def record_event(stream):
    print("🎯 Event detected, recording...")
    frames = []

    for _ in range(int(SAMPLE_RATE / CHUNK_SIZE * 2)):  # ~2 sec
        chunk, _ = stream.read(CHUNK_SIZE)
        frames.append(chunk)

    audio = np.concatenate(frames)

    return audio

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE) as stream:

    print("🎤 Listening continuously...")
    while True:
        chunk, _ = stream.read(CHUNK_SIZE)
        chunk = chunk.flatten()
        energy = compute_energy(chunk)

        if energy > THRESHOLD:
            audio = record_event(stream)
            sf.write("temp.wav", audio, SAMPLE_RATE)
            with open("temp.wav", "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            client.publish("soniq/audio", encoded)

            print("📡 Sent event")