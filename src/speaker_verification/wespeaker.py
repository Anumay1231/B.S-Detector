"""
wespeaker.py
------------
Second speaker encoder for Phase 9 score fusion: WeSpeaker ResNet34-LM
(Wespeaker/wespeaker-voxceleb-resnet34-LM, VoxCeleb2-trained, large-margin
fine-tuned), run through its official ONNX export with onnxruntime on CPU.

Why a second model: on held-out Hindi and Indian-English speakers (Google SVQ)
ResNet34 alone is about as accurate as ECAPA-TDNN, but its errors fall on
different pairs, so averaging the two normalised scores removes 10-30% of the
errors ECAPA makes alone (docs/speaker-verification-hindi-english.md).

Input: 16 kHz mono float32 in [-1, 1]. Output: a 256-dim embedding.
Features follow WeSpeaker's recipe: 80-dim Kaldi fbank, 25 ms / 10 ms,
no dither, input scaled to 16-bit range, per-utterance mean normalisation.
"""

from __future__ import annotations

import numpy as np

MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"


class WeSpeakerEncoder:
    def __init__(self, repo: str = MODEL_REPO, filename: str = MODEL_FILE) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(hf_hub_download(repo, filename), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        x = torch.as_tensor(np.asarray(waveform, dtype=np.float32)).reshape(1, -1) * (1 << 15)
        feats = kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=16000)
        feats = feats - feats.mean(dim=0, keepdim=True)
        return self.session.run(None, {self.input_name: feats.unsqueeze(0).numpy()})[0][0]
