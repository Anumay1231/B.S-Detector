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
    SpeakerMetadataError,
    TrialListParseError,
    load_speaker_name_map,
    normalize_utterance_id,
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


# --------------------------------------------------------------------------
# load_speaker_name_map (vox1_meta.csv)
# --------------------------------------------------------------------------


def _write_meta(tmp_path, body: str):
    """Write a synthetic vox1_meta.csv (tab-separated, with the official
    header row) and return its path."""
    meta = tmp_path / "vox1_meta.csv"
    meta.write_text(
        "VoxCeleb1 ID\tVGGFace1 ID\tGender\tNationality\tSet\n" + body,
        encoding="utf-8",
    )
    return meta


def test_load_speaker_name_map_parses_official_format(tmp_path) -> None:
    meta = _write_meta(
        tmp_path,
        "id10001\tA.J._Buckley\tm\tIreland\tdev\n"
        "id10270\tEartha_Kitt\tf\tUSA\ttest\n",
    )
    mapping = load_speaker_name_map(str(meta))
    assert mapping == {"A.J._Buckley": "id10001", "Eartha_Kitt": "id10270"}


def test_load_speaker_name_map_skips_header_and_blank_lines(tmp_path) -> None:
    meta = _write_meta(tmp_path, "\nid10270\tEartha_Kitt\tf\tUSA\ttest\n\n")
    assert load_speaker_name_map(str(meta)) == {"Eartha_Kitt": "id10270"}


def test_load_speaker_name_map_strips_padded_fields(tmp_path) -> None:
    """Some mirrors re-save the file with padded columns."""
    meta = _write_meta(tmp_path, "id10270 \t Eartha_Kitt \t f \t USA \t test\n")
    assert load_speaker_name_map(str(meta)) == {"Eartha_Kitt": "id10270"}


def test_load_speaker_name_map_accepts_whitespace_separated(tmp_path) -> None:
    meta = tmp_path / "vox1_meta.csv"
    meta.write_text(
        "VoxCeleb1 ID VGGFace1 ID Gender Nationality Set\n"
        "id10270 Eartha_Kitt f USA test\n",
        encoding="utf-8",
    )
    assert load_speaker_name_map(str(meta)) == {"Eartha_Kitt": "id10270"}


def test_load_speaker_name_map_conflicting_ids_raises(tmp_path) -> None:
    meta = _write_meta(
        tmp_path,
        "id10270\tEartha_Kitt\tf\tUSA\ttest\n"
        "id10999\tEartha_Kitt\tf\tUSA\ttest\n",
    )
    with pytest.raises(SpeakerMetadataError):
        load_speaker_name_map(str(meta))


def test_load_speaker_name_map_no_usable_rows_raises(tmp_path) -> None:
    meta = _write_meta(tmp_path, "")
    with pytest.raises(SpeakerMetadataError):
        load_speaker_name_map(str(meta))


def test_load_speaker_name_map_missing_file_raises() -> None:
    with pytest.raises(SpeakerMetadataError):
        load_speaker_name_map("/does/not/exist/vox1_meta.csv")


# --------------------------------------------------------------------------
# normalize_utterance_id
# --------------------------------------------------------------------------


def test_normalize_leaves_canonical_ids_unchanged() -> None:
    canonical = "id10270/x6uYqmx31kE/00001.wav"
    assert normalize_utterance_id(canonical) == canonical
    assert normalize_utterance_id(canonical, {"Eartha_Kitt": "id10270"}) == canonical


def test_normalize_converts_speaker_name_form() -> None:
    assert (
        normalize_utterance_id(
            "Eartha_Kitt/x6uYqmx31kE_0000001.wav", {"Eartha_Kitt": "id10270"}
        )
        == "id10270/x6uYqmx31kE/00001.wav"
    )


def test_normalize_handles_video_ids_containing_underscores() -> None:
    """VoxCeleb video ids can contain underscores, so the split must be
    on the LAST underscore, not the first."""
    assert (
        normalize_utterance_id(
            "Eartha_Kitt/5sJomL_D0_g_0000005.wav", {"Eartha_Kitt": "id10270"}
        )
        == "id10270/5sJomL_D0_g/00005.wav"
    )


def test_normalize_repads_utterance_number_rather_than_truncating() -> None:
    assert (
        normalize_utterance_id("Name/vid_0000055.wav", {"Name": "id10270"})
        == "id10270/vid/00055.wav"
    )
    assert (
        normalize_utterance_id("Name/vid_0012345.wav", {"Name": "id10270"})
        == "id10270/vid/12345.wav"
    )


def test_normalize_name_form_without_metadata_raises() -> None:
    with pytest.raises(TrialListParseError, match="vox1_meta.csv"):
        normalize_utterance_id("Eartha_Kitt/x6uYqmx31kE_0000001.wav")


def test_normalize_unknown_speaker_name_raises() -> None:
    with pytest.raises(TrialListParseError, match="not present"):
        normalize_utterance_id("Unknown_Person/vid_0000001.wav", {"Eartha_Kitt": "id10270"})


def test_normalize_non_numeric_utterance_number_raises() -> None:
    with pytest.raises(TrialListParseError):
        normalize_utterance_id("Name/vid_notanumber.wav", {"Name": "id10270"})


def test_normalize_missing_underscore_raises() -> None:
    with pytest.raises(TrialListParseError):
        normalize_utterance_id("Name/novideoseparator.wav", {"Name": "id10270"})


def test_normalize_unrecognized_shape_raises() -> None:
    with pytest.raises(TrialListParseError):
        normalize_utterance_id("just_a_filename.wav", {"Name": "id10270"})


def test_normalize_backslash_paths_are_handled() -> None:
    assert (
        normalize_utterance_id("Name\\vid_0000001.wav", {"Name": "id10270"})
        == "id10270/vid/00001.wav"
    )


# --------------------------------------------------------------------------
# parse_trial_list with speaker-name-form lists
# --------------------------------------------------------------------------


def test_parse_trial_list_name_form_with_metadata(tmp_path) -> None:
    meta = _write_meta(
        tmp_path,
        "id10270\tEartha_Kitt\tf\tUSA\ttest\n"
        "id10271\tEd_Westwick\tm\tUK\ttest\n",
    )
    trial_list = tmp_path / "voxceleb1_test.txt"
    trial_list.write_text(
        "1 Eartha_Kitt/x6uYqmx31kE_0000001.wav Eartha_Kitt/8jEAjG6SegY_0000008.wav\n"
        "0 Eartha_Kitt/x6uYqmx31kE_0000001.wav Ed_Westwick/ize_eiCFEg0_0000003.wav\n"
    )

    trials = parse_trial_list(
        str(trial_list), audio_root="/audio", meta_path=str(meta)
    )

    assert len(trials) == 2

    genuine = trials[0]
    assert genuine.label == 1
    assert genuine.reference_id == "id10270/x6uYqmx31kE/00001.wav"
    assert genuine.test_id == "id10270/8jEAjG6SegY/00008.wav"
    assert genuine.reference_speaker == genuine.test_speaker == "id10270"

    impostor = trials[1]
    assert impostor.label == 0
    assert impostor.test_id == "id10271/ize_eiCFEg0/00003.wav"
    assert impostor.reference_speaker == "id10270"
    assert impostor.test_speaker == "id10271"


def test_parse_trial_list_name_form_resolves_paths_under_audio_root(tmp_path) -> None:
    import os

    meta = _write_meta(tmp_path, "id10270\tEartha_Kitt\tf\tUSA\ttest\n")
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text(
        "1 Eartha_Kitt/vidA_0000001.wav Eartha_Kitt/vidB_0000002.wav\n"
    )

    trials = parse_trial_list(
        str(trial_list), audio_root="/audio_root", meta_path=str(meta)
    )

    assert trials[0].reference_path == os.path.join(
        "/audio_root", "id10270", "vidA", "00001.wav"
    )


def test_parse_trial_list_name_form_without_meta_raises_with_line_number(tmp_path) -> None:
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text(
        "1 id10270/vidA/00001.wav id10270/vidB/00002.wav\n"
        "1 Eartha_Kitt/vidA_0000001.wav Eartha_Kitt/vidB_0000002.wav\n"
    )
    with pytest.raises(TrialListParseError, match=r"trials\.txt:2"):
        parse_trial_list(str(trial_list), audio_root="/audio")


def test_parse_trial_list_canonical_form_still_works_with_meta_supplied(tmp_path) -> None:
    """Passing --meta must not break ordinary numeric-id trial lists."""
    meta = _write_meta(tmp_path, "id10270\tEartha_Kitt\tf\tUSA\ttest\n")
    trial_list = tmp_path / "trials.txt"
    trial_list.write_text("1 id10270/vidA/00001.wav id10270/vidB/00002.wav\n")

    trials = parse_trial_list(
        str(trial_list), audio_root="/audio", meta_path=str(meta)
    )

    assert trials[0].reference_id == "id10270/vidA/00001.wav"
    assert trials[0].reference_speaker == "id10270"
