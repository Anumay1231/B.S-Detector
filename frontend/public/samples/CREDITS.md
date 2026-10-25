# Sample audio credits

All clips were converted to 16 kHz mono 16-bit PCM WAV, trimmed to about 0.15 s of
leading and trailing silence, and peak-normalised to about -1 dBFS. Nothing else was
changed.

## real-hindi.wav: Google FLEURS

- Dataset: FLEURS, `hi_in` config, dev split, sentence id 1537, file `14112322580063897119.wav` (female speaker).
- Citation: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia,
  Jason Riesa, Clara Rivera, Ankur Bapna. "FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech." IEEE SLT 2022. arXiv:2205.12446.
- License: Creative Commons Attribution 4.0 International (CC-BY-4.0).
- Source: https://huggingface.co/datasets/google/fleurs

## speaker-a-1.wav, speaker-a-2.wav, speaker-b-1.wav: LibriSpeech

| File | Utterance | Reader (speaker id) |
| --- | --- | --- |
| speaker-a-1.wav | 777-126732-0023 | 777 |
| speaker-a-2.wav | 777-126732-0027 | 777 |
| speaker-b-1.wav | 84-121123-0003 | 84 |

- Corpus: LibriSpeech ASR corpus, dev-clean (Hugging Face `openslr/librispeech_asr`,
  config `clean`, split `validation`). The recordings are LibriVox public-domain audiobook
  readings by volunteer readers.
- Citation: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur. "LibriSpeech: an
  ASR corpus based on public domain audio books." ICASSP 2015.
- License: Creative Commons Attribution 4.0 International (CC-BY-4.0).
- Source: https://www.openslr.org/12

## fake-hindi-xtts.wav: synthetic speech

- Generated with Coqui XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`, coqui-tts 0.28.0)
  using the model's built-in stock studio speaker "Ana Florence", language `hi`, speaking the
  same sentence as `real-hindi.wav`.
- No real person's voice was cloned: no reference clip was given to the model.
- The XTTS-v2 model is distributed under the Coqui Public Model License (CPML), which permits
  non-commercial use only. https://coqui.ai/cpml
