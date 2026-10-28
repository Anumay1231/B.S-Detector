# B.S. Detector

### [▶ Download and run B.S. Detector](https://github.com/Anumay1231/B.S-Detector/archive/refs/heads/main.zip)

Unzip it, then double-click `start.bat` (Windows) or `start.command` (macOS). Full steps are [below](#run-it).

A voice authentication system for Hindi and English speech that asks two questions about a recording, in order:

1. **Is this voice real?** A deepfake detector, a frozen **XLS-R 300M** model with a **Sensitive Layer Selection (SLS)** head, listens for signs that the speech was generated.
2. **Is it the same person?** If both voices sound genuine, two pretrained speaker models (**ECAPA-TDNN** and **WeSpeaker ResNet34**) compare it with a known recording of the claimed speaker, against a cut-off calibrated on Hindi and Indian-English speakers.

The two stages are joined by an **interlock**. If stage 1 calls a voice synthetic, stage 2 never runs, because a convincing clone would otherwise pass as the real person. The repository contains the research behind both stages and a local web app that runs them on your own computer.

### Run it

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
  Stage 2     │ ECAPA + ResNet34 per  │        │ ECAPA + ResNet34 per  │
  speaker     │ 3 s piece, averaged   │        │ 3 s piece, averaged   │
  verification└───────────┬───────────┘        └───────────┬───────────┘
                          └─── fused cosine similarity ─────┘
                                         │
              cut-off rises with speech length; too close → "not sure"
```

**Stage 1** turns 4.04 seconds of 16 kHz audio into the 24 hidden layers of XLS-R 300M. A learned sigmoid gate weighs each layer, the layers are blended and pooled, and two dense layers output the chance that the voice is synthetic. Only the 23.4M-parameter head is trained; the 315M backbone stays frozen.

**Stage 2** uses `speechbrain/spkrec-ecapa-voxceleb` and `Wespeaker/wespeaker-voxceleb-resnet34-LM` as published. Each clip's speech is cut into 3 s pieces, each piece becomes a voiceprint, and the voiceprints are averaged. Each model's cosine similarity is normalised and the two are averaged. The cut-off grows with the amount of speech in the shorter clip, and scores just around it come back "not sure". The constants were fitted on Hindi and English speakers (Common Voice, Google SVQ) rather than chosen by hand. The interlock checks the known voice for a deepfake too.

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

VoxCeleb1 is mostly English celebrity interviews, so these figures are an **English baseline**. Details: [`docs/calibration.md`](docs/calibration.md).

### Stage 2: speaker verification on Hindi and Indian English (Phase 9)

On held-out Google SVQ speakers, the VoxCeleb cut-off accepted 42-70% of different same-gender speakers, which is why two friends on one laptop came out as one person. With score fusion, segment averaging, a duration-aware cut-off and a "not sure" band (2-fold cross-validation over speakers, at least 8 s of speech per clip):

| Held-out SVQ | Clean | Same room (simulated) |
| :-- | --: | --: |
| Hindi: accuracy on answers / answered | **100.0%** / 98% | **96.9%** / 81% |
| Indian English: accuracy on answers / answered | **98.1%** / 95% | **99.0%** / 93% |
| Old VoxCeleb cut-off, accuracy (Hindi / English) | 65.7% / 73.4% | 64.8% / 74.2% |

Details, data and every intermediate step: [`docs/speaker-verification-hindi-english.md`](docs/speaker-verification-hindi-english.md). Demo clips for both stages: [`demo/`](demo/).

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
models/               detector heads and the fine-tuned ECAPA (in Git; see Limitations)
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

- Phone-call audio has only been tested on a simulated line (8 kHz, mu-law): `python scripts/ui_path_check.py --phone` gets 7 answers wrong. Three English fakes get past stage 1, and the speaker check says "not sure" on the same-room pairs. Real phone calls and compressed codecs are untested.
- The app uses the silence-invariant Hindi head (`models/sls_head_silence_invariant.pth`), then the IndicSynth head, then the channel-robust head. All three are in the repo, and the status tag says "Hindi head". If none is there, `run.py` downloads the silence-invariant head from GitHub; without it the app falls back to the English checkpoint, which misses most Hindi fakes.
- Stage 1 gets all 27 demo and sample files right in the browser's window, which starts at 0 s. Scored over the whole file, though, it flags 8 of the 18 real ones, all Google SVQ phone recordings, so it only judges the first 4.04 s for now. Run `python scripts/ui_path_check.py` against a running server to see the current state, and see [`docs/deepfake-improvement-plan.md`](docs/deepfake-improvement-plan.md).
- Stage 2 needs about 8-15 s of speech in each clip to reach the accuracy above; shorter clips are marked less reliable. Hinglish (code-mixed) speech has not been tested separately.
- Stage 2 needs the voice clearly above the background. The app trims silence before comparing and refuses clips whose speech sits within 6 dB of the noise: ECAPA-TDNN embeds a noise-dominated clip as the room rather than the person, and in tests on random LibriSpeech speakers buried in noise, most pairs of different people crossed the threshold.

## Credits

Stage 1 checkpoint: [sukhdeveyash/XLS-R-SLS-Deepfake-Detection](https://huggingface.co/sukhdeveyash/XLS-R-SLS-Deepfake-Detection). Backbone: [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m). Stage 2 models: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), [Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM). Phase 9 data: [Common Voice 17](https://huggingface.co/datasets/fixie-ai/common_voice_17_0), [Google SVQ](https://huggingface.co/datasets/google/svq). Data: [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof), [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

B.S. Detector is a research tool. Its verdicts are evidence, not proof.
