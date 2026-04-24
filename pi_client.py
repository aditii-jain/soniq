import sounddevice as sd
import soundfile as sf
import paho.mqtt.client as mqtt
import base64

SAMPLE_RATE = 16000
DURATION = 2

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883)

def record_audio():
    print("🎤 Recording...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio

while True:
    input("Press Enter to record...")

    audio = record_audio()
    sf.write("temp.wav", audio, SAMPLE_RATE)

    with open("temp.wav", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    client.publish("soniq/audio", encoded)
    print("📡 Sent audio")