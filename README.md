# B.S. Detector

A voice authentication system for Hindi and English speech that asks two questions about a recording, in order:

1. **Is this voice real?** A deepfake detector, a frozen **XLS-R 300M** model with a **Sensitive Layer Selection (SLS)** head, listens for signs that the speech was generated.
2. **Is it the same person?** If the voice sounds genuine, a pretrained **ECAPA-TDNN** speaker model compares it with a known recording of the claimed speaker.

The two stages are joined by an **interlock**. If stage 1 calls a voice synthetic, stage 2 never runs, because a convincing clone would otherwise pass as the real person. The repository contains the research behind both stages and a local web app that runs them on your own computer.

### [▶ Download and run B.S. Detector](https://github.com/Anumay1231/B.S-Detector/archive/refs/heads/main.zip)

1. Unzip the download.
2. Double-click **`start.bat`** (Windows) or **`start.command`** (macOS). On Linux, run `./start.command`.
3. Your browser opens at `http://127.0.0.1:8000` when the app is ready.

Or from a terminal:

```bash
git clone https://github.com/Anumay1231/B.S-Detector && cd B.S-Detector && python run.py
```

You need **Python 3.10 or newer**. The first run creates a `.venv`, installs PyTorch, SpeechBrain and the other dependencies, and downloads the model weights from Hugging Face: about 1.3 GB for XLS-R and 80 MB for ECAPA-TDNN. Later starts take a few seconds. Node.js isn't needed, because the built interface ships in `frontend/dist`.

## How it works

```
                    Voice to check                    Known voice
                          │                                │
              ┌───────────▼───────────┐                    │
  Stage 1     │ XLS-R 300M (frozen)   │                    │
  deepfake    │ 24 layers → SLS head  │                    │
  detection   └───────────┬───────────┘                    │
                          │                                │
               synthetic? ├──── yes ──► interlock: stop,   │
                          │             identity not checked
                          no                               │
                          │                                │
              ┌───────────▼───────────┐        ┌───────────▼───────────┐
  Stage 2     │ ECAPA-TDNN voiceprint │        │ ECAPA-TDNN voiceprint │
  speaker     └───────────┬───────────┘        └───────────┬───────────┘
  verification            └──────── cosine similarity ─────┘
                                         │
                            ≥ 0.2528 → same speaker
```

**Stage 1** turns 4.04 seconds of 16 kHz audio into the 24 hidden layers of XLS-R 300M. A learned sigmoid gate weighs each layer, the layers are blended and pooled, and two dense layers output the chance that the voice is synthetic. Only the 23.4M-parameter head is trained; the 315M backbone stays frozen.

**Stage 2** uses `speechbrain/spkrec-ecapa-voxceleb` exactly as published. It turns each whole clip into a 192-dimensional voiceprint, and the two voiceprints are compared with cosine similarity. The cut-off, 0.252784, was calibrated on VoxCeleb1 trials rather than chosen by hand.

## Results

### Stage 1: deepfake detection on Hindi

Tested on a balanced set the model had never seen: 200 real voices (IndicTTS studio speech and Mozilla Common Voice) and 200 fakes, 40 from each of Edge-TTS, VITS-MMS, Indic-TTS, FastSpeech and XTTS-v2. The starting checkpoint was trained on English; the adapted rows retrain only the SLS head on Hindi clips.

| Setup | Fakes caught (of 200) | Real voices passed (of 200) | Accuracy | EER |
| :-- | --: | --: | --: | --: |
| English-trained, no Hindi | 48 | 199 | 61.75% | 32.25% |
| Adapted on 10 Hindi clips per class | 59 | 198 | 64.33 ± 5.04% | 27.64 ± 2.27% |
| Adapted on 50 Hindi clips per class | 117 | 186 | 75.83 ± 10.49% | 18.18 ± 1.18% |
| **Adapted on 500 Hindi clips per class** | **191** | **196** | **96.83 ± 0.82%** | **2.66 ± 0.52%** |

Adapted rows are the mean of 3 seeds; the counts are the mean of their confusion matrices. The best N=500 run reached 98.00% accuracy and 2.12% EER, with 6 fakes missed and 2 false alarms out of 400. In earlier runs on a 439-clip split, unfreezing the top of XLS-R made things worse: 24.14% EER with 2 layers unfrozen and 25.31% with 4, where one run let 162 of 189 fakes through. Details: [`docs/deepfake-detection.md`](docs/deepfake-detection.md), the [technical dossier](docs/HINDI_DEEPFAKE_DETECTION_TECHNICAL_DOSSIER.pdf) and the JSON in [`outputs/`](outputs/).

### Stage 2: speaker verification on VoxCeleb1

| Run | Trials | EER | Threshold | Notes |
| :-- | --: | --: | --: | :-- |
| Speaker-disjoint split | 28,280 | 0.87% | **0.252784** | Sets the threshold the app uses. On 20 held-out speakers: 1.15% of impostors accepted, 0.48% of genuine pairs rejected |
| Full official protocol | 37,720 | 1.04% | 0.272217 | For comparison with published results (about 0.80% on the cleaned list). ROC-AUC 0.9989 |

VoxCeleb1 is mostly English celebrity interviews, so these figures are an **English baseline**. Stage 2 has not yet been validated on Hindi or Hinglish speech. Details: [`docs/calibration.md`](docs/calibration.md).

## The web app

- **Check one voice:** upload a file (WAV, MP3, FLAC, OGG, M4A) or record from your microphone. The page draws the spectrogram, lets you drag the 4.04-second window the detector hears, and shows the verdict with the gate weight of each of the 24 layers.
- **Compare two voices:** load a known voice and a voice to check. The deepfake check runs first, and the voices are compared only if the voice to check sounds genuine.
- **Findings and method:** the results above, explained on the page.
- **Session log:** every check you run, exportable as JSON. Nothing leaves your computer.

The server is FastAPI (`server/app.py`) and exposes `GET /api/status`, `POST /api/analyze` (stage 1) and `POST /api/verify` (both stages, with the interlock). Interactive API docs are at `/api/docs` while it runs.

## Repository layout

```
run.py, start.bat, start.command   one-command and double-click launchers
server/
  app.py              FastAPI app: API and the built frontend
  inference.py        stage 1 inference (XLS-R 300M + SLS head)
  speaker.py          stage 2 adapter over src/speaker_verification
frontend/             React + TypeScript (Vite); dist/ is the built app
src/speaker_verification/
                      stage 2 package: audio, ECAPA-TDNN encoder, similarity,
                      calibration, verifier, VoxCeleb datasets
scripts/              stage 1 benchmark pipeline and stage 2 command-line tools
tests/                stage 2 unit tests (pytest)
docs/                 deepfake dossier, experiment protocol, calibration and model notes
outputs/              stage 1 benchmark results (JSON) and figures
models/               local SLS head checkpoints (not in Git; see Limitations)
data/                 dataset placeholders; downloaded audio is never committed
```

## Reproducing the research

The research environment is separate from the app. `requirements.txt` lists both stages' research dependencies, and `requirements-dev.txt` adds pytest.

**Stage 1.** Stream the Hindi SEA-Spoof subset, cache XLS-R features and run the zero-shot and few-shot experiments:

```bash
python scripts/run_deepfake_detection.py --device cuda --test-per-class 200 --fewshot-sizes 10 50 500
```

Use `--skip-streaming` and `--skip-features` to reuse caches. `scripts/generate_paper_figures.py` rebuilds the figures.

**Stage 2.** Run the unit tests, then score a pair of files with the calibrated threshold:

```bash
python -m pytest
python scripts/verify.py reference.wav test.wav --threshold 0.252784 --calibrated
```

The VoxCeleb calibration pipeline (`build_trials.py`, `extract_embeddings.py`, `score_trials.py`, `run_voxceleb_calibration.py`) and the full module documentation are in [`docs/speaker-verification.md`](docs/speaker-verification.md).

## Developing the app

```bash
python run.py --no-browser                     # API on :8000
cd frontend && npm install && npm run dev      # UI on :5173, proxies /api to :8000
npm run build                                  # rebuild frontend/dist before committing
```

`BSD_DEVICE` (`cuda`, `mps` or `cpu`) picks the device for both models. `BSD_CHECKPOINT` (`v1` or `v2`) picks the English stage 1 checkpoint, and `BSD_HEAD` points at a different Hindi head file. `BSD_SV_THRESHOLD` overrides the stage 2 cut-off, and the app then labels it uncalibrated.

## Limitations

- Both stages were evaluated on clean recordings. Phone-call audio (8 kHz, compressed codecs) has not been tested.
- The Hindi head (N=500) is not in Git because of its size (89 MB). Put `sls_head_n500_seed2_best.pth` in `models/` and the app uses it automatically; the status tag then says "Hindi head". Without it the app falls back to the English checkpoint, which passes real Hindi voices but misses most Hindi fakes.
- Stage 1 hears 4.04 seconds at a time. Check several windows of a long clip.
- The stage 2 threshold is an English (VoxCeleb1) baseline and has not been validated on Hindi or Hinglish speech.
- Stage 2 needs the voice clearly above the background. The app trims silence before comparing and refuses clips whose speech sits within 6 dB of the noise: ECAPA-TDNN embeds a noise-dominated clip as the room rather than the person, and in tests on random LibriSpeech speakers buried in noise, most pairs of different people crossed the threshold.

## Credits

Stage 1 checkpoint: [sukhdeveyash/XLS-R-SLS-Deepfake-Detection](https://huggingface.co/sukhdeveyash/XLS-R-SLS-Deepfake-Detection). Backbone: [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m). Stage 2 model: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). Data: [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof), [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

B.S. Detector is a research tool. Its verdicts are evidence, not proof.
