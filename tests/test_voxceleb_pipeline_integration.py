"""
test_voxceleb_pipeline_integration.py
-----------------------------------------

Small, fully synthetic end-to-end test of the Phase 7 pipeline:

    parse_trial_list -> build_subset -> write_trials_csv
        -> collect_unique_utterances -> extract_embeddings (stub encoder)
        -> score_trials -> run_calibration

using a handful of generated WAV files and a hand-written trial-list
file (VoxCeleb's official text format) -- no real VoxCeleb download and
no real ECAPA-TDNN model are required. This exercises the same
plumbing scripts/build_trials.py -> scripts/extract_embeddings.py ->
scripts/score_trials.py -> scripts/run_voxceleb_calibration.py use,
without needing network access or the pretrained model (which this
sandbox cannot reach -- see docs/model.md).
"""

from __future__ import annotations

import csv
import math
import struct
import wave

from build_trials import build_subset, write_trials_csv
from extract_embeddings import collect_unique_utterances, extract_embeddings
from run_voxceleb_calibration import run_calibration
from score_trials import score_trials
from speaker_verification.datasets.voxceleb import parse_trial_list
from speaker_verification.embedding_cache import EmbeddingCache


def _write_tiny_wav(path, frequency_hz: float) -> None:
    """Write a very short, valid mono 16 kHz WAV file. Content doesn't
    matter -- the stub encoder below derives a fake "embedding" from
    the file path, not the audio -- this just needs to be a real file
    that exists on disk."""
    sample_rate = 16000
    n_samples = 800
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            value = int(3000 * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
            frames += struct.pack("<h", value)
        wf.writeframes(bytes(frames))


class _DeterministicStubEncoder:
    """A stand-in for SpeakerEncoder that produces a deterministic,
    speaker-dependent fake embedding from the utterance path, so
    genuine (same speaker) pairs score high and impostor (different
    speaker) pairs score low -- without running the real model."""

    def encode_file(self, path: str):
        import torch

        # Speaker id is the parent-of-parent directory name, matching
        # the VoxCeleb <speaker>/<video>/<utt>.wav layout.
        import os

        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = sum(ord(c) for c in speaker_id)
        base = torch.tensor([math.sin(seed + i) for i in range(8)])
        return base / base.norm()


def test_full_synthetic_voxceleb_pipeline(tmp_path) -> None:
    audio_root = tmp_path / "audio"
    speakers = [f"id{100 + i}" for i in range(6)]  # 6 speakers
    for speaker in speakers:
        video_dir = audio_root / speaker / "vid1"
        video_dir.mkdir(parents=True)
        for utt in ("00001.wav", "00002.wav"):
            _write_tiny_wav(video_dir / utt, frequency_hz=220.0)

    # Hand-write a VoxCeleb-format trial list: genuine pairs (same
    # speaker, two utterances) and impostor pairs (different speakers).
    trial_lines = []
    for speaker in speakers:
        trial_lines.append(f"1 {speaker}/vid1/00001.wav {speaker}/vid1/00002.wav")
    for a, b in zip(speakers, speakers[1:] + speakers[:1]):
        trial_lines.append(f"0 {a}/vid1/00001.wav {b}/vid1/00001.wav")
    trial_list_path = tmp_path / "trial_list.txt"
    trial_list_path.write_text("\n".join(trial_lines) + "\n")

    # --- build_trials ---
    all_trials = parse_trial_list(str(trial_list_path), audio_root=str(audio_root))
    result = build_subset(
        all_trials, max_genuine=None, max_impostor=None, seed=1,
        calibration_fraction=0.5, split_seed=1,
    )
    assert result["missing"] == []
    trials_csv = tmp_path / "trials.csv"
    write_trials_csv(str(trials_csv), result["calibration"], result["evaluation"])

    # --- extract_embeddings ---
    unique = collect_unique_utterances(str(trials_csv))
    assert len(unique) > 0
    cache = EmbeddingCache(str(tmp_path / "cache"))
    cache.load()
    stats = extract_embeddings(
        unique, _DeterministicStubEncoder(), cache,
        model_source="stub-model", sample_rate=16000, save_every=0,
    )
    assert stats["num_failed"] == 0
    assert stats["num_processed"] == len(unique)

    # --- score_trials ---
    with open(trials_csv, newline="", encoding="utf-8") as f:
        trial_rows = list(csv.DictReader(f))
    scored, skipped = score_trials(trial_rows, cache)
    assert skipped == []
    assert len(scored) == len(trial_rows)

    # --- calibration ---
    if result["calibration"] and result["evaluation"]:
        cal_has_both = (
            any(r["label"] == "1" for r in scored if r["split"] == "calibration")
            and any(r["label"] == "0" for r in scored if r["split"] == "calibration")
        )
        if cal_has_both:
            calibration_output = run_calibration(
                [{**r, "label": int(r["label"]), "score": float(r["score"])} for r in scored]
            )
            cal = calibration_output["calibration_result"]
            assert 0.0 <= cal.eer <= 1.0
            # Our deterministic stub encoder makes genuine scores == 1.0
            # (identical fake embedding for same speaker) and impostor
            # scores < 1.0 (different fake embeddings) -- so this
            # synthetic setup should calibrate to near-perfect
            # separation.
            assert cal.eer <= 0.5
