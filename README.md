# B.S. Detector

Check whether a voice clip is real or synthetic. Drop in a recording and a frozen **XLS-R 300M** speech model with a **selective layer summarization (SLS)** head gives a verdict. It also shows the 4-second window it heard and which of its 24 layers it relied on. Everything runs on your own computer.

### [▶ Download and run B.S. Detector](https://github.com/Anumay1231/B.S-Detector/archive/refs/heads/main.zip)

1. Unzip the download.
2. Double-click **`start.bat`** (Windows) or **`start.command`** (macOS). On Linux, run `./start.command`.
3. Your browser opens at `http://127.0.0.1:8000` when the app is ready.

Or from a terminal:

```bash
git clone https://github.com/Anumay1231/B.S-Detector && cd B.S-Detector && python run.py
```

You need **Python 3.9 or newer**. The first run creates a `.venv`, installs PyTorch and the other dependencies, and downloads about 1.3 GB of model weights from Hugging Face. Later starts take a few seconds. Node.js isn't needed: the built interface ships in `frontend/dist`.

## What the app does

- **Analyze:** upload a file (WAV, MP3, FLAC, OGG, M4A) or record from your microphone. The browser converts the audio to 16 kHz mono and draws its spectrogram. You can drag the 4.04-second analysis window to any part of the clip.
- **Verdict:** the chance that the window is synthetic, plus the learned gate weight for each of XLS-R's 24 layers.
- **Findings:** the Hindi benchmark results behind the model, including where it fails.
- **Session log:** every check you run, exportable as JSON.

## How well it works

We tested on a balanced Hindi set of 200 real voices (IndicTTS, Mozilla Common Voice) and 200 fakes, with 40 from each of five generators: Edge-TTS, VITS-MMS, Indic-TTS, FastSpeech and XTTS-v2.

| Setup | Fakes caught (of 200) | Real voices passed (of 200) | Accuracy | EER |
| :-- | --: | --: | --: | --: |
| English-trained, no Hindi | 44 | 196 | 60.00% | 28.90% |
| Adapted on 10 Hindi clips per class | 74 | 189 | 65.83 ± 5.87% | 26.88 ± 1.83% |
| Adapted on 50 Hindi clips per class | 155 | 164 | 79.67 ± 0.47% | 19.70 ± 0.57% |

Unfreezing the top layers of XLS-R made things worse: 24.14% EER with 2 layers unfrozen and 25.31% with 4. The app runs the English-trained checkpoint, so on Hindi speech a "genuine" verdict is weak evidence. Full numbers are in [`outputs/`](outputs/) and [`paper/`](paper/).

## Project layout

```
run.py              one-command launcher (venv, dependencies, server, browser)
start.bat           double-click launcher for Windows
start.command       double-click launcher for macOS / Linux
server/
  app.py            FastAPI: /api/status, /api/analyze, serves the frontend
  inference.py      XLS-R 300M + SLS head inference
frontend/           React + TypeScript (Vite); dist/ is the built app
scripts/            benchmark and training pipeline
outputs/            benchmark results (JSON) and figures
paper/              paper draft and research dossier
```

## Developing the interface

```bash
python run.py --no-browser        # API on :8000
cd frontend && npm install && npm run dev   # UI on :5173, proxies /api to :8000
npm run build                     # rebuild frontend/dist before committing
```

`BSD_DEVICE` (`cuda`, `mps` or `cpu`) and `BSD_CHECKPOINT` (`v1` or `v2`) override the defaults.

## Credits

SLS checkpoint: [sukhdeveyash/XLS-R-SLS-Deepfake-Detection](https://huggingface.co/sukhdeveyash/XLS-R-SLS-Deepfake-Detection). Backbone: [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m). Data: [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof).

This is a research tool. Its verdicts are evidence, not proof.
