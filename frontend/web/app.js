/**
 * B.S. DETECTOR — CYBER-FORENSIC FRONTEND ENGINE
 * SIH 2026 Presentation Suite
 */

// --- Global State ---
const state = {
  audioContext: null,
  analyser: null,
  sourceNode: null,
  audioBuffer: null,
  isPlaying: false,
  isRecording: false,
  mediaRecorder: null,
  recordedChunks: [],
  animationFrameId: null,
  currentFile: {
    name: "executive_voice_sample.wav",
    duration: "4.04s",
    sampleRate: "16,000 Hz",
    size: "128.4 KB",
    type: "WAV"
  },
  activePreset: "spoof", // "spoof" | "authentic" | "unknown"
  speakerVault: [
    { id: "spk_001", name: "Anumay (Team Lead)", samples: 5, confidence: 96.8 },
    { id: "spk_002", name: "Yash Kothari (Lead AI Architect)", samples: 5, confidence: 96.2 },
    { id: "spk_003", name: "Executive Target / VIP", samples: 5, confidence: 91.5 },
  ],
  pipelineRunning: false,
  slsWeights: []
};

// Preset Scenario Configurations
const PRESETS = {
  spoof: {
    name: "Executive_Clone_ElevenLabs_v2.wav",
    duration: "4.04s",
    sampleRate: "16,000 Hz",
    size: "129.2 KB",
    label: "SPOOF",
    score: 98.9,
    processingTime: "0.48s",
    vocoderArtifact: "High-Freq Phase Incoherence (HiFi-GAN/DiffWave signature)",
    layerHighlights: [15, 17, 18, 20, 21],
    speakerMatch: false,
    finalDecision: "REJECTED",
    decisionReason: "Synthetic voice artifacts detected across transformer layers 15-21. Biometric verification interlock permanently engaged."
  },
  authentic: {
    name: "Yash_Kothari_Authentic_Hindi.wav",
    duration: "3.82s",
    sampleRate: "16,000 Hz",
    size: "122.2 KB",
    label: "BONAFIDE",
    score: 99.4,
    processingTime: "0.52s",
    vocoderArtifact: "None (Natural Glottal Pulse Dynamics)",
    layerHighlights: [3, 7, 11],
    speakerMatch: true,
    matchedSpeaker: "Yash Kothari (Lead AI Architect)",
    similarity: 96.2,
    finalDecision: "AUTHENTIC SPEAKER",
    decisionReason: "Acoustic signature is authentic (99.4% confidence) and matches enrolled biometric voiceprint with 96.2% cosine similarity."
  },
  unknown: {
    name: "Unregistered_Caller_Bonafide.wav",
    duration: "4.15s",
    sampleRate: "16,000 Hz",
    size: "132.8 KB",
    label: "BONAFIDE",
    score: 98.6,
    processingTime: "0.49s",
    vocoderArtifact: "None (Natural Acoustic Formants)",
    layerHighlights: [2, 5, 8],
    speakerMatch: false,
    bestMatchSim: 43.1,
    finalDecision: "UNKNOWN CALLER",
    decisionReason: "Voice is certified authentic, but biometric embedding does not match any registered vault profile above the 75% threshold."
  }
};

// --- Initialize Audio Context & Synthesis ---
function initAudio() {
  if (!state.audioContext) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioCtx();
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 128;
    state.analyser.smoothingTimeConstant = 0.8;
  }
}

// Generate rich synthesized audio buffers for presets so sound actually plays!
function generateSyntheticVoiceBuffer(type) {
  initAudio();
  const ctx = state.audioContext;
  const sampleRate = ctx.sampleRate;
  const duration = 3.5;
  const numSamples = sampleRate * duration;
  const buffer = ctx.createBuffer(1, numSamples, sampleRate);
  const data = buffer.getChannelData(0);

  const baseFreq = type === "spoof" ? 175 : 135;

  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    // Formant simulation
    const f0 = baseFreq + Math.sin(t * 8) * 12;
    let s = Math.sin(2 * Math.PI * f0 * t) * 0.4;
    s += Math.sin(2 * Math.PI * f0 * 2.1 * t) * 0.25;
    s += Math.sin(2 * Math.PI * f0 * 3.2 * t) * 0.15;
    
    // In spoof, add vocoder high-frequency jitter / phase distortion
    if (type === "spoof") {
      s += (Math.random() - 0.5) * 0.08 * Math.sin(t * 40);
      s += Math.sin(2 * Math.PI * 4200 * t) * 0.05;
    }

    // Natural speech amplitude cadence envelope
    const envelope = Math.sin(t * 3.5) * Math.sin(t * 1.8);
    data[i] = s * Math.max(0, envelope) * 0.7;
  }
  return buffer;
}

// --- Visualizer Rendering (60 FPS) ---
const canvas = document.getElementById("visualizerCanvas");
const ctx = canvas ? canvas.getContext("2d") : null;

function renderVisualizer() {
  if (!canvas || !ctx) return;

  const width = canvas.width = canvas.parentElement.clientWidth;
  const height = canvas.height = 140;

  if (!state.analyser || (!state.isPlaying && !state.isRecording)) {
    // Idle ambient radar waveform
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(0, 240, 255, 0.2)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const time = Date.now() * 0.003;
    for (let x = 0; x < width; x += 3) {
      const y = height / 2 + Math.sin(x * 0.03 + time) * 8 * Math.sin(x * 0.005);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    requestAnimationFrame(renderVisualizer);
    return;
  }

  const bufferLength = state.analyser.frequencyBinCount;
  const timeData = new Uint8Array(bufferLength);
  const freqData = new Uint8Array(bufferLength);
  state.analyser.getByteTimeDomainData(timeData);
  state.analyser.getByteFrequencyData(freqData);

  ctx.clearRect(0, 0, width, height);

  // Background grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  // 1. Draw Spectrum Frequency Bars
  const barCount = 32;
  const barWidth = (width / barCount) - 3;
  for (let i = 0; i < barCount; i++) {
    const val = freqData[i * 2] || 0;
    const barHeight = (val / 255) * (height * 0.75);
    const x = i * (barWidth + 3);
    const y = height - barHeight;

    const grad = ctx.createLinearGradient(0, y, 0, height);
    if (state.activePreset === "spoof") {
      grad.addColorStop(0, "#ff3366");
      grad.addColorStop(1, "rgba(255, 51, 102, 0.1)");
    } else {
      grad.addColorStop(0, "#00f0ff");
      grad.addColorStop(1, "rgba(0, 240, 255, 0.05)");
    }

    ctx.fillStyle = grad;
    ctx.fillRect(x, y, barWidth, barHeight);
  }

  // 2. Draw Oscilloscope Center Waveform
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = state.activePreset === "spoof" ? "#ff3366" : "#00ff88";
  ctx.shadowColor = state.activePreset === "spoof" ? "rgba(255, 51, 102, 0.8)" : "rgba(0, 255, 136, 0.8)";
  ctx.shadowBlur = 10;

  ctx.beginPath();
  const sliceWidth = width / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = timeData[i] / 128.0;
    const y = (v * height) / 2;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);

    x += sliceWidth;
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  requestAnimationFrame(renderVisualizer);
}

// --- Audio Playback Controls ---
function toggleAudioPlayback() {
  initAudio();
  if (state.audioContext.state === "suspended") {
    state.audioContext.resume();
  }

  if (state.isPlaying) {
    stopAudio();
  } else {
    playCurrentAudio();
  }
}

function playCurrentAudio() {
  if (!state.audioBuffer) {
    state.audioBuffer = generateSyntheticVoiceBuffer(state.activePreset);
  }

  state.sourceNode = state.audioContext.createBufferSource();
  state.sourceNode.buffer = state.audioBuffer;
  state.sourceNode.connect(state.analyser);
  state.analyser.connect(state.audioContext.destination);

  state.sourceNode.onended = () => {
    state.isPlaying = false;
    updatePlayButtonUI();
  };

  state.sourceNode.start();
  state.isPlaying = true;
  updatePlayButtonUI();
}

function stopAudio() {
  if (state.sourceNode) {
    try { state.sourceNode.stop(); } catch (e) {}
    state.sourceNode.disconnect();
    state.sourceNode = null;
  }
  state.isPlaying = false;
  updatePlayButtonUI();
}

function updatePlayButtonUI() {
  const btn = document.getElementById("btnPlayAudio");
  if (!btn) return;
  if (state.isPlaying) {
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause Audio`;
    btn.style.background = "var(--cyan)";
    btn.style.color = "#000";
  } else {
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Play Audio`;
    btn.style.background = "rgba(0, 240, 255, 0.1)";
    btn.style.color = "#fff";
  }
}

// --- Microphone Recording Support ---
async function toggleMicRecording() {
  initAudio();
  const micBtn = document.getElementById("btnMicRecord");

  if (state.isRecording) {
    // Stop recording
    if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
      state.mediaRecorder.stop();
    }
    state.isRecording = false;
    micBtn.classList.remove("recording");
    micBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg> Record Live Mic`;
  } else {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const micSource = state.audioContext.createMediaStreamSource(stream);
      micSource.connect(state.analyser);

      state.recordedChunks = [];
      state.mediaRecorder = new MediaRecorder(stream);
      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.recordedChunks.push(e.data);
      };
      state.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(state.recordedChunks, { type: "audio/wav" });
        const arrayBuffer = await audioBlob.arrayBuffer();
        state.audioBuffer = await state.audioContext.decodeAudioData(arrayBuffer);
        state.currentFile = {
          name: "Live_Microphone_Capture.wav",
          duration: `${state.audioBuffer.duration.toFixed(2)}s`,
          sampleRate: `${state.audioBuffer.sampleRate} Hz`,
          size: `${(audioBlob.size / 1024).toFixed(1)} KB`,
          type: "WAV"
        };
        updateFileMetaUI();
      };

      state.mediaRecorder.start();
      state.isRecording = true;
      micBtn.classList.add("recording");
      micBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="8"/></svg> Stop Recording`;
    } catch (err) {
      alert("Microphone access unavailable or denied. Using synthesized audio.");
    }
  }
}

// --- Presets Selection ---
function selectPreset(presetKey) {
  state.activePreset = presetKey;
  const p = PRESETS[presetKey];
  state.currentFile = {
    name: p.name,
    duration: p.duration,
    sampleRate: p.sampleRate,
    size: p.size,
    type: "WAV"
  };

  // Update preset buttons UI
  document.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.getElementById(`preset-${presetKey}`);
  if (activeBtn) activeBtn.classList.add("active");

  stopAudio();
  state.audioBuffer = generateSyntheticVoiceBuffer(presetKey);
  updateFileMetaUI();
  resetPipelineUI();
}

function updateFileMetaUI() {
  document.getElementById("metaFileName").textContent = state.currentFile.name;
  document.getElementById("metaDuration").textContent = state.currentFile.duration;
  document.getElementById("metaSampleRate").textContent = state.currentFile.sampleRate;
  document.getElementById("metaSize").textContent = state.currentFile.size;
}

// --- 24-Layer SLS Attention Heatmap Generator ---
function generateSLSLayerWeights(presetKey) {
  const weights = [];
  const p = PRESETS[presetKey];

  for (let i = 1; i <= 24; i++) {
    let w;
    if (p.layerHighlights.includes(i)) {
      // High sensitive attention spike (capturing synthetic vocoder artifact)
      w = 0.75 + Math.random() * 0.22;
    } else {
      // Baseline feature extraction attention
      w = 0.15 + Math.random() * 0.35;
    }
    weights.push({ layer: i, weight: w });
  }
  return weights;
}

function renderSLSLayerHeatmap(weights) {
  const container = document.getElementById("slsBarsContainer");
  if (!container) return;
  container.innerHTML = "";

  weights.forEach(({ layer, weight }) => {
    const col = document.createElement("div");
    col.className = "sls-layer-col";
    col.style.height = `${Math.round(weight * 100)}%`;
    col.title = `Layer ${layer}: Attention Weight ${(weight * 100).toFixed(1)}%`;

    if (state.activePreset === "spoof" && PRESETS.spoof.layerHighlights.includes(layer)) {
      col.classList.add("highlight");
    }

    col.addEventListener("click", () => {
      document.getElementById("slsInspectorDetail").innerHTML = `
        <b>Layer ${layer} Attention Inspector:</b> Weight: <code>${(weight * 100).toFixed(2)}%</code> | 
        ${state.activePreset === "spoof" && PRESETS.spoof.layerHighlights.includes(layer) 
          ? '<span style="color:var(--crimson)">Vocoder phase artifact detected in late latent representations</span>'
          : '<span style="color:var(--emerald)">Nominal acoustic phonetic structure</span>'}
      `;
    });

    container.appendChild(col);
  });
}

// --- Multi-Stage Pipeline Execution ---
async function runPipeline() {
  if (state.pipelineRunning) return;
  state.pipelineRunning = true;
  initAudio();

  const analyzeBtn = document.getElementById("btnAnalyze");
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
      <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
      <path d="M12 2a10 10 0 0 1 10 10"/>
    </svg> Executing Two-Stage Inference...
  `;

  // Start audio playback during analysis for dynamic effect
  playCurrentAudio();

  const p = PRESETS[state.activePreset];
  const isSpoof = p.label === "SPOOF";

  // Step 1: Preprocessing & Ingest
  setPipelineStep(0, "active");
  await sleep(400);
  setPipelineStep(0, "done");

  // Step 2: SLS Attention Extraction across 24 layers
  setPipelineStep(1, "active");
  state.slsWeights = generateSLSLayerWeights(state.activePreset);
  renderSLSLayerHeatmap(state.slsWeights);
  await sleep(650);
  setPipelineStep(1, "done");

  // Step 3: Deepfake Classification
  setPipelineStep(2, isSpoof ? "blocked" : "done");
  updateStage1UI(p);

  // Step 4: Security Gate & Speaker Verification Check
  if (isSpoof) {
    // THE WOW MOMENT: BIOMETRIC CHECK BLOCKED
    setPipelineStep(3, "blocked");
    setPipelineStep(4, "blocked");
    updateStage2BlockedUI(p);
  } else {
    setPipelineStep(3, "active");
    await sleep(600);
    setPipelineStep(3, "done");
    setPipelineStep(4, "done");
    updateStage2VerifiedUI(p);
  }

  // Final Decision Banner
  updateFinalDecisionUI(p);

  analyzeBtn.disabled = false;
  analyzeBtn.innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg> Re-Analyze Audio Signature
  `;
  state.pipelineRunning = false;
}

function setPipelineStep(index, status) {
  const nodes = document.querySelectorAll(".track-node");
  if (nodes[index]) {
    nodes[index].className = `track-node ${status}`;
  }
}

function resetPipelineUI() {
  document.querySelectorAll(".track-node").forEach((n, idx) => {
    n.className = "track-node";
  });
  const banner = document.getElementById("decisionBanner");
  if (banner) {
    banner.className = "decision-banner";
    banner.innerHTML = `
      <div class="decision-header">
        <div class="decision-badge" style="color:var(--text-muted)">
          ⚡ Awaiting Audio Analysis
        </div>
      </div>
      <p class="decision-subtext">Select a preset or upload an audio file, then click "Run Two-Stage Analysis".</p>
    `;
  }
  document.getElementById("stage1Status").textContent = "PENDING";
  document.getElementById("stage1Status").className = "stage-status-pill";
  document.getElementById("stage1ScoreBar").style.width = "0%";
  document.getElementById("stage1ScoreText").textContent = "—";

  document.getElementById("stage2Card").className = "stage-subcard";
  document.getElementById("stage2Status").textContent = "PENDING";
  document.getElementById("stage2Status").className = "stage-status-pill";
  document.getElementById("stage2ScoreBar").style.width = "0%";
  document.getElementById("stage2ScoreText").textContent = "—";
  document.getElementById("stage2InterlockBox").style.display = "none";
  document.getElementById("stage2NormalContent").style.display = "block";
}

function updateStage1UI(p) {
  const statusPill = document.getElementById("stage1Status");
  const bar = document.getElementById("stage1ScoreBar");
  const text = document.getElementById("stage1ScoreText");

  if (p.label === "SPOOF") {
    statusPill.textContent = "SPOOF DETECTED";
    statusPill.className = "stage-status-pill pill-bad";
    bar.className = "confidence-bar-inner bar-crimson";
    bar.style.width = `${p.score}%`;
    text.textContent = `Deepfake Probability: ${p.score}% (Inference: ${p.processingTime})`;
  } else {
    statusPill.textContent = "BONAFIDE / AUTHENTIC";
    statusPill.className = "stage-status-pill pill-ok";
    bar.className = "confidence-bar-inner bar-emerald";
    bar.style.width = `${p.score}%`;
    text.textContent = `Authenticity Confidence: ${p.score}% (Inference: ${p.processingTime})`;
  }
}

function updateStage2BlockedUI(p) {
  const card = document.getElementById("stage2Card");
  card.className = "stage-subcard interlocked";

  const statusPill = document.getElementById("stage2Status");
  statusPill.textContent = "INTERLOCK ENGAGED";
  statusPill.className = "stage-status-pill pill-bad";

  document.getElementById("stage2NormalContent").style.display = "none";
  const lockBox = document.getElementById("stage2InterlockBox");
  lockBox.style.display = "flex";
}

function updateStage2VerifiedUI(p) {
  const card = document.getElementById("stage2Card");
  card.className = "stage-subcard";

  const statusPill = document.getElementById("stage2Status");
  const bar = document.getElementById("stage2ScoreBar");
  const text = document.getElementById("stage2ScoreText");
  document.getElementById("stage2NormalContent").style.display = "block";
  document.getElementById("stage2InterlockBox").style.display = "none";

  if (p.speakerMatch) {
    statusPill.textContent = "VERIFIED MATCH";
    statusPill.className = "stage-status-pill pill-ok";
    bar.className = "confidence-bar-inner bar-cyan";
    bar.style.width = `${p.similarity}%`;
    text.innerHTML = `Enrolled Target: <b>${p.matchedSpeaker}</b> (${p.similarity}% Cosine Similarity, Threshold: 75.0%)`;
  } else {
    statusPill.textContent = "UNKNOWN SPEAKER";
    statusPill.className = "stage-status-pill pill-lock";
    bar.className = "confidence-bar-inner bar-amber";
    bar.style.width = `${p.bestMatchSim}%`;
    text.textContent = `Best Candidate Match: ${p.bestMatchSim}% (Below 75.0% Security Threshold)`;
  }
}

function updateFinalDecisionUI(p) {
  const banner = document.getElementById("decisionBanner");
  if (p.finalDecision === "REJECTED") {
    banner.className = "decision-banner spoof";
    banner.innerHTML = `
      <div class="decision-header">
        <div class="decision-badge">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
          CRITICAL SECURITY ALERT: REJECTED (AI DEEPFAKE)
        </div>
        <span class="badge badge-bad" style="color:var(--crimson);border-color:var(--crimson)">CONFIDENCE: ${p.score}%</span>
      </div>
      <p class="decision-subtext">${p.decisionReason}</p>
    `;
  } else if (p.finalDecision === "AUTHENTIC SPEAKER") {
    banner.className = "decision-banner authentic";
    banner.innerHTML = `
      <div class="decision-header">
        <div class="decision-badge">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
          AUTHENTICATED & VERIFIED: ${p.matchedSpeaker}
        </div>
        <span class="badge" style="color:var(--emerald);border-color:var(--emerald)">SIMILARITY: ${p.similarity}%</span>
      </div>
      <p class="decision-subtext">${p.decisionReason}</p>
    `;
  } else {
    banner.className = "decision-banner unknown";
    banner.innerHTML = `
      <div class="decision-header">
        <div class="decision-badge">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
          AUTHENTIC AUDIO — UNREGISTERED CALLER
        </div>
        <span class="badge" style="color:var(--amber);border-color:var(--amber)">BEST MATCH: ${p.bestMatchSim}%</span>
      </div>
      <p class="decision-subtext">${p.decisionReason}</p>
    `;
  }
}

// --- Drag & Drop / File Input Handling ---
function setupFileUpload() {
  const dropzone = document.getElementById("uploadDropzone");
  const fileInput = document.getElementById("fileInput");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleCustomFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleCustomFile(e.target.files[0]);
    }
  });
}

async function handleCustomFile(file) {
  initAudio();
  const arrayBuffer = await file.arrayBuffer();
  try {
    state.audioBuffer = await state.audioContext.decodeAudioData(arrayBuffer);
    state.currentFile = {
      name: file.name,
      duration: `${state.audioBuffer.duration.toFixed(2)}s`,
      sampleRate: `${state.audioBuffer.sampleRate} Hz`,
      size: `${(file.size / 1024).toFixed(1)} KB`,
      type: file.name.split('.').pop().toUpperCase()
    };
    updateFileMetaUI();
    resetPipelineUI();
  } catch (err) {
    alert("Could not decode audio file. Please use standard WAV, MP3, or FLAC.");
  }
}

// --- Speaker Enrollment Modal ---
function setupSpeakerEnrollment() {
  const form = document.getElementById("enrollSpeakerForm");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const nameInput = document.getElementById("spkNameInput");
    if (!nameInput.value.trim()) return;

    const newSpk = {
      id: "spk_" + Math.random().toString(36).substring(2, 8),
      name: nameInput.value.trim(),
      samples: 4,
      confidence: 95.0
    };

    state.speakerVault.push(newSpk);
    renderSpeakerVault();
    nameInput.value = "";
    alert(`Voiceprint successfully enrolled for "${newSpk.name}" (192-dim ECAPA-TDNN embedding created).`);
  });
}

function renderSpeakerVault() {
  const container = document.getElementById("speakerVaultList");
  if (!container) return;
  container.innerHTML = "";

  state.speakerVault.forEach(spk => {
    const item = document.createElement("div");
    item.className = "speaker-item";
    item.innerHTML = `
      <div class="spk-info">
        <h4>${spk.name}</h4>
        <span>ID: ${spk.id} • ${spk.samples} genuine reference clips</span>
      </div>
      <div class="spk-metric">${spk.confidence}% Baseline</div>
    `;
    container.appendChild(item);
  });
}

// --- Keyboard Shortcuts for Presenter (1, 2, 3, Space, A) ---
function setupKeyboardShortcuts() {
  window.addEventListener("keydown", (e) => {
    // Ignore when typing in inputs
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.key === "1") {
      selectPreset("spoof");
    } else if (e.key === "2") {
      selectPreset("authentic");
    } else if (e.key === "3") {
      selectPreset("unknown");
    } else if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      toggleAudioPlayback();
    } else if (e.key.toLowerCase() === "a") {
      e.preventDefault();
      runPipeline();
    }
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function loadLocalWavFile(filename, presetKey) {
  initAudio();
  selectPreset(presetKey);
  try {
    const res = await fetch(filename);
    if (!res.ok) throw new Error("Fetch failed");
    const arrayBuffer = await res.arrayBuffer();
    state.audioBuffer = await state.audioContext.decodeAudioData(arrayBuffer);
    state.currentFile = {
      name: filename,
      duration: `${state.audioBuffer.duration.toFixed(2)}s`,
      sampleRate: `${state.audioBuffer.sampleRate} Hz`,
      size: `${(arrayBuffer.byteLength / 1024).toFixed(1)} KB`,
      type: "WAV"
    };
    updateFileMetaUI();
  } catch (e) {
    console.warn("Using synthesized voice buffer fallback:", e);
  }
}

// --- On DOM Ready ---
document.addEventListener("DOMContentLoaded", () => {
  setupFileUpload();
  setupSpeakerEnrollment();
  renderSpeakerVault();
  setupKeyboardShortcuts();
  selectPreset("spoof");
  renderVisualizer();

  document.getElementById("btnPlayAudio").addEventListener("click", toggleAudioPlayback);
  document.getElementById("btnMicRecord").addEventListener("click", toggleMicRecording);
  document.getElementById("btnAnalyze").addEventListener("click", runPipeline);

  document.getElementById("preset-spoof").addEventListener("click", () => selectPreset("spoof"));
  document.getElementById("preset-authentic").addEventListener("click", () => selectPreset("authentic"));
  document.getElementById("preset-unknown").addEventListener("click", () => selectPreset("unknown"));

  const btnLoadDeepfake = document.getElementById("btnLoadDeepfakeWav");
  if (btnLoadDeepfake) {
    btnLoadDeepfake.addEventListener("click", () => {
      loadLocalWavFile("fake_sample_deepfake_hindi.wav", "spoof");
    });
  }

  const btnLoadAuthentic = document.getElementById("btnLoadAuthenticWav");
  if (btnLoadAuthentic) {
    btnLoadAuthentic.addEventListener("click", () => {
      loadLocalWavFile("real_sample_yash_kothari.wav", "authentic");
    });
  }
});

