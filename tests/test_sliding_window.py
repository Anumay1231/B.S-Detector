"""tests/test_sliding_window.py
-------------------------------
Unit tests for full-clip sliding window extraction, batching, and aggregation.
"""

import numpy as np
import pytest
import torch

from server.inference import (
    INPUT_LENGTH,
    TARGET_SR,
    extract_sliding_windows,
    ForensicInferenceResult,
    SLSHead,
)


def test_sliding_window_short_audio():
    # Audio under 4.04s should tile-pad to 1 window
    sr = 16000
    waveform = torch.randn(int(2.0 * sr)) * 0.1
    windows, meta = extract_sliding_windows(waveform, window_length=INPUT_LENGTH, stride_sec=2.0, sample_rate=sr)

    assert len(windows) == 1
    assert len(meta) == 1
    assert windows[0].shape[0] == INPUT_LENGTH
    assert meta[0]["index"] == 0
    assert meta[0]["is_padded"] is True
    assert meta[0]["end_sec"] == 2.0


def test_sliding_window_exact_length():
    sr = 16000
    waveform = torch.randn(INPUT_LENGTH) * 0.1
    windows, meta = extract_sliding_windows(waveform, window_length=INPUT_LENGTH, stride_sec=2.0, sample_rate=sr)

    assert len(windows) == 1
    assert len(meta) == 1
    assert windows[0].shape[0] == INPUT_LENGTH
    assert meta[0]["is_padded"] is False


def test_sliding_window_multi_window_coverage():
    sr = 16000
    # 8.0 seconds of audio with 2.0s stride
    waveform = torch.randn(int(8.0 * sr)) * 0.1
    windows, meta = extract_sliding_windows(waveform, window_length=INPUT_LENGTH, stride_sec=2.0, sample_rate=sr)

    assert len(windows) >= 3
    assert len(meta) == len(windows)
    for w in windows:
        assert w.shape[0] == INPUT_LENGTH

    # Verify that first window starts at 0.0s and last window reaches 8.0s
    assert meta[0]["start_sec"] == 0.0
    assert meta[-1]["end_sec"] == 8.0


def test_sliding_window_max_windows_cap():
    sr = 16000
    # 60 seconds of audio
    waveform = torch.randn(int(60.0 * sr)) * 0.05
    max_win = 10
    windows, meta = extract_sliding_windows(
        waveform, window_length=INPUT_LENGTH, stride_sec=1.0, sample_rate=sr, max_windows=max_win
    )

    assert len(windows) <= max_win
    assert len(meta) <= max_win
    assert meta[-1]["end_sec"] == 60.0


def test_sls_head_batched_forward():
    head = SLSHead(hidden_dim=1024)
    head.eval()

    K = 5
    hidden_states = [torch.randn(K, 201, 1024) for _ in range(24)]

    with torch.no_grad():
        log_probs, weights = head(hidden_states)

    assert log_probs.shape == (K, 2)
    assert len(weights) == 24
    assert weights[0].shape == (K, 1)

    probs = torch.exp(log_probs)
    # Each row sums to ~1.0
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(K), atol=1e-5)


def test_detection_json_sliding_window_structure():
    from server.app import detection_json

    result = ForensicInferenceResult(
        verdict="SPOOF",
        confidence=98.5,
        prob_bonafide=0.015,
        prob_spoof=0.985,
        layer_weights=[0.5] * 24,
        latency_ms=120.0,
        audio_duration_sec=10.0,
        device_used="cpu",
        interlock_triggered=True,
        layer_summary="Synthetic phase anomalies detected.",
        windows=[
            {"index": 0, "start_sec": 0.0, "end_sec": 4.04, "prob_spoof": 0.05, "prob_bonafide": 0.95, "verdict": "BONAFIDE"},
            {"index": 1, "start_sec": 2.0, "end_sec": 6.04, "prob_spoof": 0.985, "prob_bonafide": 0.015, "verdict": "SPOOF"},
            {"index": 2, "start_sec": 4.0, "end_sec": 8.04, "prob_spoof": 0.970, "prob_bonafide": 0.030, "verdict": "SPOOF"},
        ],
        num_windows=3,
        aggregation="sliding_window_max",
        peak_spoof_window={"index": 1, "start_sec": 2.0, "end_sec": 6.04, "prob_spoof": 0.985, "prob_bonafide": 0.015, "verdict": "SPOOF"}
    )

    data = detection_json(result)
    assert data["verdict"] == "SPOOF"
    assert data["probSpoof"] == 0.985
    assert data["numWindows"] == 3
    assert data["aggregation"] == "sliding_window_max"
    assert len(data["windows"]) == 3
    assert data["peakSpoofWindow"]["start_sec"] == 2.0
    assert data["peakSpoofWindow"]["prob_spoof"] == 0.985

