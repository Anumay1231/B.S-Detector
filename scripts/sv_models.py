"""
Alternative pretrained speaker encoders for scripts/sv_benchmark.py.

Each takes a 16 kHz mono float32 waveform (already through the app's
speech_only trimming) and returns one embedding vector. Used only to compare
against the deployed ECAPA-TDNN; none of them is wired into the app.
"""

from __future__ import annotations

import numpy as np


def _hf(repo: str, name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name)


class WeSpeaker:
    """WeSpeaker ResNet (VoxCeleb2 + large-margin fine-tune), ONNX export."""

    def __init__(self, repo: str, onnx: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(_hf(repo, onnx), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        x = torch.from_numpy(wav).unsqueeze(0) * (1 << 15)
        feats = kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=16000)
        feats = feats - feats.mean(dim=0, keepdim=True)
        return self.sess.run(None, {self.inp: feats.unsqueeze(0).numpy()})[0][0]


class ResNet34(WeSpeaker):
    name = "wespeaker_r34"

    def __init__(self):
        super().__init__("Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx")


class ResNet293(WeSpeaker):
    name = "wespeaker_r293"

    def __init__(self):
        super().__init__("Wespeaker/wespeaker-voxceleb-resnet293-LM", "voxceleb_resnet293_LM.onnx")


class WavLM:
    name = "wavlm_sv"

    def __init__(self):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        torch.set_grad_enabled(False)
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        inputs = self.fe(wav, sampling_rate=16000, return_tensors="pt")
        return self.model(**inputs).embeddings[0].numpy()


class Ecapa2:
    name = "ecapa2"

    def __init__(self):
        import torch

        torch.set_grad_enabled(False)
        self.model = torch.jit.load(_hf("Jenthe/ECAPA2", "ecapa2.pt"), map_location="cpu").float().eval()

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        import torch

        return self.model(torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()


MODELS = {m.name: m for m in (ResNet34, ResNet293, WavLM, Ecapa2)}
