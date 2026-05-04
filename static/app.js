const alertsForm = document.getElementById("alerts-form");
const alertsMessage = document.getElementById("alerts-message");
const nameCheckbox = document.getElementById("name-checkbox");
const nameRecorderSection = document.getElementById("name-recorder-section");
const startRecordingBtn = document.getElementById("start-recording-btn");
const stopRecordingBtn = document.getElementById("stop-recording-btn");
const recordingStatus = document.getElementById("recording-status");
const latestDetection = document.getElementById("latest-detection");
const introSection = document.getElementById("intro");
const consoleSection = document.getElementById("console");

let audioContext = null;
let mediaStream = null;
let source = null;
let processor = null;
let pcmChunks = [];
const sampleRate = 16000;

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Uint8Array(buffer);
}

function encodeWav(samples, sr) {
  const pcmBytes = floatTo16BitPCM(samples);
  const wavBuffer = new ArrayBuffer(44 + pcmBytes.length);
  const view = new DataView(wavBuffer);

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + pcmBytes.length, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, pcmBytes.length, true);

  new Uint8Array(wavBuffer, 44).set(pcmBytes);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function updateNameSectionVisibility() {
  nameRecorderSection.classList.toggle("hidden", !nameCheckbox.checked);
}

alertsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selected = Array.from(
    document.querySelectorAll('input[name="alertType"]:checked')
  ).map((el) => el.value);

  const res = await fetch("/api/alerts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alerts: selected }),
  });

  if (!res.ok) {
    alertsMessage.textContent = "Failed to save preferences.";
    return;
  }
  alertsMessage.textContent = "Preferences saved.";
});

nameCheckbox.addEventListener("change", updateNameSectionVisibility);

startRecordingBtn.addEventListener("click", async () => {
  recordingStatus.textContent = "Requesting microphone access...";
  pcmChunks = [];

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext({ sampleRate });
    source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
      const channelData = event.inputBuffer.getChannelData(0);
      pcmChunks.push(new Float32Array(channelData));
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    startRecordingBtn.disabled = true;
    stopRecordingBtn.disabled = false;
    recordingStatus.textContent = "Recording...";
  } catch (error) {
    recordingStatus.textContent = "Microphone access failed.";
  }
});

stopRecordingBtn.addEventListener("click", async () => {
  stopRecordingBtn.disabled = true;
  startRecordingBtn.disabled = false;
  recordingStatus.textContent = "Saving name.wav...";

  if (processor) {
    processor.disconnect();
  }
  if (source) {
    source.disconnect();
  }
  if (audioContext) {
    await audioContext.close();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  const totalLength = pcmChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  pcmChunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });

  const wavBlob = encodeWav(merged, sampleRate);
  const formData = new FormData();
  formData.append("audio", wavBlob, "name.wav");

  const res = await fetch("/api/name-recording", { method: "POST", body: formData });
  if (res.ok) {
    recordingStatus.textContent = "Saved name.wav successfully.";
  } else {
    recordingStatus.textContent = "Failed to save name.wav.";
  }
});

// ── Notification helpers ──────────────────────────────────────

async function requestNotificationPermission() {
  if (!("Notification" in window)) return;
  if (Notification.permission === "default") {
    await Notification.requestPermission();
  }
}

function fireNotification(sound, score) {
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  const scoreSuffix = typeof score === "number" ? ` (${score.toFixed(2)})` : "";
  new Notification("Soniq Alert", {
    body: `Detected: ${sound}${scoreSuffix}`,
    icon: "https://cdn-icons-png.flaticon.com/512/727/727218.png",
    tag: "soniq-detection",
    renotify: true,
  });
}

function getSelectedAlerts() {
  return Array.from(document.querySelectorAll('input[name="alertType"]:checked')).map(
    (el) => el.value
  );
}

function updateDetectionUI(detection) {
  if (!detection || !detection.sound) return;
  const scoreSuffix =
    typeof detection.score === "number" ? ` (${detection.score.toFixed(2)})` : "";
  latestDetection.textContent = `${detection.sound}${scoreSuffix}`;
}

// ── Server-Sent Events ────────────────────────────────────────

function connectSSE() {
  const es = new EventSource("/api/events");

  es.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    if (msg.type !== "detection") return;

    const { detection, alerts } = msg;
    updateDetectionUI(detection);

    const selected = getSelectedAlerts();
    const soundKey = (detection.sound || "").toLowerCase();
    const shouldAlert =
      selected.length === 0 ||
      selected.some((pref) => soundKey.includes(pref));

    if (shouldAlert) {
      fireNotification(detection.sound, detection.score);
    }
  };

  es.onerror = () => {
    es.close();
    // Reconnect after 3 s if the connection drops.
    setTimeout(connectSSE, 3000);
  };
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    updateDetectionUI(data.last_detection);
  } catch (err) {
    // Keep UI quiet if backend is temporarily unavailable.
  }
}

function smoothScrollToConsole() {
  if (!consoleSection) return;

  const startY = window.scrollY;
  const targetY = consoleSection.offsetTop;
  const distance = targetY - startY;
  if (Math.abs(distance) < 8) return;

  const durationMs = 1800;
  const startTs = performance.now();

  const step = (now) => {
    const elapsed = now - startTs;
    const progress = Math.min(1, elapsed / durationMs);
    const eased = 1 - (1 - progress) ** 3;
    window.scrollTo(0, startY + distance * eased);
    if (progress < 1) requestAnimationFrame(step);
  };

  requestAnimationFrame(step);
}

function setupAutoIntroScroll() {
  if (!introSection || !consoleSection) return;
  window.scrollTo(0, 0);
  window.setTimeout(() => {
    smoothScrollToConsole();
  }, 650);
}

updateNameSectionVisibility();
setupAutoIntroScroll();
refreshStatus();          
connectSSE();             
requestNotificationPermission();
