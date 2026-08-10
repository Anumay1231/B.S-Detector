"""
test_datasets_voxceleb.py
----------------------------

Unit tests for src/speaker_verification/datasets/voxceleb.py: parsing
the official VoxCeleb trial-list text format into Trial objects.

Uses small, hand-written synthetic trial-list files (matching the
documented "<label> <ref> <test>" format) -- no real VoxCeleb data is
downloaded or required.
"""

from __future__ import annotations

import pytest

from speaker_verification.datasets.trials import DatasetError
from speaker_verification.datasets.voxceleb import (
    TrialListParseError,
    parse_trial_list,
    resolve_utterance_path,
    speaker_id_from_utterance_id,
)


# --------------------------------------------------------------------------
# speaker_id_from_utterance_id
# --------------------------------------------------------------------------


def test_speaker_id_extraction() -> None:
    assert speaker_id_from_utterance_id("id10270/x6uYqmx31kE/00001.wav") == "id10270"


def test_speaker_id_extraction_missing_prefix_raises() -> None:
    with pytest.raises(TrialListParseError):
        speaker_id_from_utterance_id("not_a_voxceleb_path.wav")


# --------------------------------------------------------------------------
# resolve_utterance_path
# --------------------------------------------------------------------------


def test_resolve_utterance_path_joins_correctly() -> None:
    import os

    resolved = resolve_utterance_path("/audio_root", "id10270/x6uYqmx31kE/00001.wav")
    assert resolved == os.path.join("/audio_root", "id10270", "x6uYqmx31kE", "00001.wav")


# --------------------------------------------------------------------------
# parse_trial_list: genuine/impostor labels (Phase 7K #2)
# --------------------------------------------------------------------------


def test_parse_trial_list_genuine_and_impostor_labels(tmp_path) -> None:
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text(
        "1 id10270/vidA/00001.wav id10270/vidB/00002.wav\n"
        "0 id10270/vidA/00001.wav id10300/vidC/00003.wav\n"
    )

    trials = parse_trial_list(str(trial_list), audio_root=str(tmp_path / "audio"))

    assert len(trials) == 2
    assert trials[0].label == 1
    assert trials[0].is_genuine is True
    assert trials[0].reference_speaker == "id10270"
    assert trials[0].test_speaker == "id10270"

    assert trials[1].label == 0
    assert trials[1].is_genuine is False
    assert trials[1].reference_speaker == "id10270"
    assert trials[1].test_speaker == "id10300"


def test_parse_trial_list_blank_lines_skipped(tmp_path) -> None:
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text(
        "1 id1/v/1.wav id1/v/2.wav\n"
        "\n"
        "   \n"
        "0 id1/v/1.wav id2/v/1.wav\n"
    )
    trials = parse_trial_list(str(trial_list), audio_root=str(tmp_path))
    assert len(trials) == 2


def test_parse_trial_list_trial_ids_are_traceable_to_line_numbers(tmp_path) -> None:
    trial_list = tmp_path / "mytrials.txt"
    trial_list.write_text("1 id1/v/1.wav id1/v/2.wav\n0 id1/v/1.wav id2/v/1.wav\n")
    trials = parse_trial_list(str(trial_list), audio_root=str(tmp_path))
    assert trials[0].trial_id == "mytrials.txt:1"
    assert trials[1].trial_id == "mytrials.txt:2"


def test_parse_trial_list_resolves_paths_under_audio_root(tmp_path) -> None:
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text("1 id1/v/1.wav id1/v/2.wav\n")
    trials = parse_trial_list(str(trial_list), audio_root="/audio")
    assert trials[0].reference_path.replace("\\", "/") == "/audio/id1/v/1.wav"
    assert trials[0].test_path.replace("\\", "/") == "/audio/id1/v/2.wav"


# --------------------------------------------------------------------------
# Malformed input handling
# --------------------------------------------------------------------------


def test_parse_trial_list_wrong_field_count_raises(tmp_path) -> None:
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text("1 id1/v/1.wav\n")  # missing the test path
    with pytest.raises(TrialListParseError):
        parse_trial_list(str(trial_list), audio_root=str(tmp_path))


def test_parse_trial_list_bad_label_raises(tmp_path) -> None:
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text("2 id1/v/1.wav id1/v/2.wav\n")
    with pytest.raises(TrialListParseError):
        parse_trial_list(str(trial_list), audio_root=str(tmp_path))


def test_parse_trial_list_missing_speaker_prefix_raises(tmp_path) -> None:
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text("1 not_a_speaker_path.wav id1/v/2.wav\n")
    with pytest.raises(TrialListParseError):
        parse_trial_list(str(trial_list), audio_root=str(tmp_path))


def test_parse_trial_list_missing_file_raises_dataset_error() -> None:
    with pytest.raises(DatasetError):
        parse_trial_list("/does/not/exist.txt", audio_root="/audio")
