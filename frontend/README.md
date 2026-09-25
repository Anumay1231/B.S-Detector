# 🛡️ B.S. Detector — Cyber-Forensic Frontend Suite

This folder provides a dedicated, production-grade frontend interface for testing real audio samples with the **B.S. Detector** architecture on `localhost`.

---

## 🚀 Quickstart (Localhost)

To launch the interactive dashboard on your local machine:

```bash
# Option 1: Using the launcher script
bash frontend/run.sh

# Option 2: Direct Streamlit command
streamlit run frontend/app.py --server.address localhost --server.port 8501
```

Once started, navigate to:
```
http://localhost:8501
```

---

## 🏗️ Architecture & Forensic Workflow

```
[ Input Audio: WAV / MP3 / Mic ]
              │
              ▼
   ┌───────────────────────┐
   │ Resample to 16 kHz    │
   │ Peak-Norm & Pad (64k) │
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │ Wav2Vec2 XLS-R 300M   │  ◄── 24 Transformer Layers (Frozen)
   └──────────┬────────────┘
              │ 24 Hidden States
              ▼
   ┌───────────────────────┐
   │ Selective Layer       │  ◄── Per-layer Sigmoid Attention Gating
   │ Summarization (SLS)   │  ◄── MaxPool2d + BatchNorm + Linear
   └──────────┬────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
[ Prob: Spoof ]  [ Prob: Bonafide ]
      │               │
  If SPOOF        If BONAFIDE
      │               │
      ▼               ▼
┌──────────────┐  ┌─────────────────────────┐
│ 🚨 INTERLOCK │  │ Stage 2: ECAPA-TDNN     │
│   ENGAGED    │  │ Biometric Verification  │
│ Biometrics   │  │ Cosine Match against    │
│   BLOCKED    │  │ Enrolled Voice Vault    │
└──────────────┘  └─────────────────────────┘
```

---

## 🌟 Key Features

1. **Real-Time PyTorch Neural Inference**:
   - Downloads & caches the official pretrained checkpoint (`sukhdeveyash/XLS-R-SLS-Deepfake-Detection`).
   - Supports hardware acceleration:
     - **CUDA** (NVIDIA GPUs)
     - **MPS** (Apple Silicon Metal Performance Shaders on macOS)
     - **CPU** fallback
2. **Multiple Audio Input Modes**:
   - **Upload**: Test custom WAV, MP3, FLAC, or OGG audio files.
   - **Microphone**: Live voice recording directly in the browser via `st.audio_input`.
   - **Preloaded Scenarios**: Quick-test synthetic vocoder attack and authentic reference speech.
3. **Forensic Layer Attention Visualizer**:
   - Renders a live bar chart displaying the learned attention weights across all 24 transformer layers of XLS-R 300M.
   - Reveals whether the model detected anomalies in shallow acoustic layers or deep phonetic layers.
4. **Hard Security Interlock**:
   - Adheres to the zero-trust forensic rule: *Never authenticate the identity of audio that fails acoustic authenticity checks.*
   - If Stage 1 detects a spoof, Stage 2 speaker verification is permanently interlocked and blocked.
5. **Enrolled Biometric Vault**:
   - Manage registered speaker profiles and reference voice embeddings.
   - Supports self-service profile registration with custom reference clips.
6. **Empirical Benchmarks Dashboard**:
   - Full publication results on the balanced SEA-Spoof Hindi test set (Zero-shot $60.00\%$, Few-shot $N=10$ $65.83\%$, Few-shot $N=50$ $79.67\%$).
7. **Session Audit Trail**:
   - Real-time logging of verdicts, confidence levels, and latency with 1-click JSON export.

---

## 📁 Folder Structure

```
frontend/
├── app.py              # Main Streamlit Cyber-Forensic Studio application
├── inference.py        # Core PyTorch inference engine (XLS-R + SLS Head)
├── requirements.txt    # Python dependencies for the frontend
├── run.sh              # Localhost execution script
├── README.md           # Documentation and usage guide
└── web/                # Standalone static Cyber-Forensic HTML/JS presentation site
    ├── index.html
    ├── app.js
    └── styles.css
```
