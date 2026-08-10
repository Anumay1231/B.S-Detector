"""
test_build_trials.py
-----------------------

Unit tests for scripts/build_trials.py's pure pipeline function
build_subset() (validate -> sample -> split) and write_trials_csv(),
using synthetic Trial objects and temp files -- no real VoxCeleb data
is required.
"""

from __future__ import annotations

import csv

from build_trials import build_subset, write_trials_csv
from speaker_verification.datasets.trials import Trial


def _trial(trial_id, ref_speaker, test_speaker, label, ref_path, test_path):
    return Trial(
        trial_id=trial_id,
        reference_id=f"{ref_speaker}/v/{trial_id}_a.wav",
        test_id=f"{test_speaker}/v/{trial_id}_b.wav",
        reference_path=ref_path,
        test_path=test_path,
        label=label,
        reference_speaker=ref_speaker,
        test_speaker=test_speaker,
    )


def test_build_subset_excludes_missing_files(tmp_path) -> None:
    present = tmp_path / "present.wav"
    present.write_bytes(b"x")

    trials = [
        _trial("g1", "id1", "id1", 1, str(present), str(present)),
        _trial("g2", "id2", "id2", 1, str(tmp_path / "gone.wav"), str(present)),  # missing ref
    ]

    result = build_subset(
        trials, max_genuine=None, max_impostor=None, seed=1,
        calibration_fraction=0.5, split_seed=1,
    )

    assert len(result["valid"]) == 1
    assert len(result["missing"]) == 1
    assert result["valid"][0].trial_id == "g1"


def test_build_subset_respects_max_counts_and_splits(tmp_path) -> None:
    present = tmp_path / "present.wav"
    present.write_bytes(b"x")

    trials = []
    for i in range(10):
        trials.append(_trial(f"g{i}", f"id{i}", f"id{i}", 1, str(present), str(present)))
    for i in range(10):
        trials.append(_trial(f"i{i}", f"idA{i}", f"idB{i}", 0, str(present), str(present)))

    result = build_subset(
        trials, max_genuine=4, max_impostor=4, seed=1,
        calibration_fraction=0.5, split_seed=1,
    )

    subset = result["subset"]
    assert sum(1 for t in subset if t.is_genuine) == 4
    assert sum(1 for t in subset if not t.is_genuine) == 4
    assert result["speaker_overlap"] == set()
    assert len(result["calibration"]) + len(result["evaluation"]) + len(result["dropped"]) == len(subset)


def test_write_trials_csv_has_expected_header_and_split_labels(tmp_path) -> None:
    present = tmp_path / "present.wav"
    present.write_bytes(b"x")
    calibration = [_trial("g1", "id1", "id1", 1, str(present), str(present))]
    evaluation = [_trial("g2", "id2", "id2", 1, str(present), str(present))]

    output_path = tmp_path / "out.csv"
    write_trials_csv(str(output_path), calibration, evaluation)

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["split"] == "calibration"
    assert rows[1]["split"] == "evaluation"
    assert set(rows[0].keys()) == {
        "trial_id", "reference_id", "test_id", "reference_path", "test_path",
        "label", "reference_speaker", "test_speaker", "split",
    }
