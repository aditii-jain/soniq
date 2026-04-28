Project Name: Soniq

Team Members:
Aditi Jain
Manasvi Pula 

-----------------------------------------------------------
Project Description:
Soniq is an assistive system designed for deaf or hard-of-hearing individuals. 
It uses a Raspberry Pi 4 with a microphone to detect environmental sounds, 
classifies them using a machine learning model, and alerts the user via an 
LCD display and a web application.

High-priority sounds such as sirens, alarms, and honks trigger visual alerts 
with a red LCD backlight.

------------------------------------------------------------
System Components:
- Raspberry Pi 4 (Audio capture + ML inference)
- Lavalier Microphone + Audio Interface
- Grove RGB LCD Display
- Flask Web Server

------------------------------------------------------------
How to Run the System:

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Flask web server on user device (phone, laptop etc.):
   python app.py

3. On Raspberry Pi, run:
   python pi_client.py

4. Open web UI:
   http://<server-ip>:5555

------------------------------------------------------------
External Libraries Used:
- Flask (Web server API)
- TensorFlow / TensorFlow Hub (YAMNet model) 
- tflite_runtime (Edge inference on Raspberry Pi)
- NumPy (signal processing)
- SoundFile (audio reading/writing)
- Requests (HTTP communication)
- librosa (audio resampling)

------------------------------------------------------------
AI / LLM Usage Disclosure:
Portions of the code and system design were developed with assistance from 
LLM-based tools (e.g., ChatGPT). All generated code was reviewed, tested, 
and modified by the team to ensure correctness and understanding.

------------------------------------------------------------
Notes:
- The system uses threshold-based triggering before ML classification 
  to improve efficiency.
- LCD color coding provides intuitive visual alerts.
