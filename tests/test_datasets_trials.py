"""
test_datasets_trials.py
-------------------------

Unit tests for src/speaker_verification/datasets/trials.py: the
dataset-agnostic Trial representation, file-existence validation,
deterministic sampling, and the speaker-disjoint calibration/evaluation
split.

All data here is synthetic (hand-constructed Trial objects / temp
files) -- no real dataset is required.
"""

from __future__ import annotations

import pytest

from speaker_verification.datasets.trials import (
    Trial,
    TrialValidationError,
    sample_trials,
    speaker_overlap,
    split_trials_by_speaker,
    summarize_trials,
    validate_trials,
)


def _make_trial(trial_id, ref_speaker, test_speaker, label, ref_id=None, test_id=None,
                 ref_path=None, test_path=None):
    return Trial(
        trial_id=trial_id,
        reference_id=ref_id or f"{ref_speaker}/vid/{trial_id}_ref.wav",
        test_id=test_id or f"{test_speaker}/vid/{trial_id}_test.wav",
        reference_path=ref_path or f"/fake/{ref_speaker}/vid/{trial_id}_ref.wav",
        test_path=test_path or f"/fake/{test_speaker}/vid/{trial_id}_test.wav",
        label=label,
        reference_speaker=ref_speaker,
        test_speaker=test_speaker,
    )


# --------------------------------------------------------------------------
# Trial construction / label validation
# --------------------------------------------------------------------------


def test_genuine_trial_requires_same_speaker() -> None:
    with pytest.raises(TrialValidationError):
        _make_trial("t1", "id1", "id2", label=1)  # genuine but different speakers


def test_impostor_trial_requires_different_speakers() -> None:
    with pytest.raises(TrialValidationError):
        _make_trial("t1", "id1", "id1", label=0)  # impostor but same speaker


def test_invalid_label_raises() -> None:
    with pytest.raises(TrialValidationError):
        _make_trial("t1", "id1", "id1", label=2)


def test_valid_genuine_and_impostor_trials_construct() -> None:
    genuine = _make_trial("t1", "id1", "id1", label=1)
    impostor = _make_trial("t2", "id1", "id2", label=0)
    assert genuine.is_genuine is True
    assert impostor.is_genuine is False


# --------------------------------------------------------------------------
# File-existence validation (Phase 7K #3)
# --------------------------------------------------------------------------


def test_validate_trials_separates_missing_files(tmp_path) -> None:
    present_ref = tmp_path / "ref.wav"
    present_test = tmp_path / "test.wav"
    present_ref.write_bytes(b"x")
    present_test.write_bytes(b"x")

    ok_trial = _make_trial(
        "ok", "id1", "id1", label=1,
        ref_path=str(present_ref), test_path=str(present_test),
    )
    missing_trial = _make_trial(
        "missing", "id2", "id2", label=1,
        ref_path=str(tmp_path / "does_not_exist_a.wav"),
        test_path=str(tmp_path / "does_not_exist_b.wav"),
    )

    valid, missing = validate_trials([ok_trial, missing_trial])
    assert valid == [ok_trial]
    assert missing == [missing_trial]


# --------------------------------------------------------------------------
# Deterministic sampling (Phase 7K #4)
# --------------------------------------------------------------------------


def _many_trials(n_genuine=20, n_impostor=20):
    trials = []
    for i in range(n_genuine):
        trials.append(_make_trial(f"g{i}", f"id{i}", f"id{i}", label=1))
    for i in range(n_impostor):
        trials.append(_make_trial(f"i{i}", f"idA{i}", f"idB{i}", label=0))
    return trials


def test_sample_trials_is_deterministic_for_same_seed() -> None:
    trials = _many_trials()
    subset_a = sample_trials(trials, max_genuine=5, max_impostor=5, seed=7)
    subset_b = sample_trials(trials, max_genuine=5, max_impostor=5, seed=7)
    assert [t.trial_id for t in subset_a] == [t.trial_id for t in subset_b]


def test_sample_trials_respects_max_counts() -> None:
    trials = _many_trials()
    subset = sample_trials(trials, max_genuine=3, max_impostor=4, seed=1)
    assert sum(1 for t in subset if t.is_genuine) == 3
    assert sum(1 for t in subset if not t.is_genuine) == 4


def test_sample_trials_different_seeds_can_differ() -> None:
    trials = _many_trials()
    subset_a = sample_trials(trials, max_genuine=5, max_impostor=5, seed=1)
    subset_b = sample_trials(trials, max_genuine=5, max_impostor=5, seed=2)
    ids_a = {t.trial_id for t in subset_a}
    ids_b = {t.trial_id for t in subset_b}
    assert ids_a != ids_b  # not guaranteed in general, but true for this fixture/seed pair


def test_sample_trials_not_simply_first_n() -> None:
    # If sampling just took the first N, the subset would always be
    # g0..g4 / i0..i4. Confirm the shuffled sample deviates from that
    # naive slice for at least one seed.
    trials = _many_trials()
    subset = sample_trials(trials, max_genuine=5, max_impostor=5, seed=1)
    naive_first_n = {f"g{i}" for i in range(5)} | {f"i{i}" for i in range(5)}
    assert {t.trial_id for t in subset} != naive_first_n


def test_sample_trials_none_means_keep_all() -> None:
    trials = _many_trials(n_genuine=3, n_impostor=4)
    subset = sample_trials(trials, max_genuine=None, max_impostor=None, seed=1)
    assert len(subset) == len(trials)


# --------------------------------------------------------------------------
# Calibration / evaluation split + speaker leakage (Phase 7K #10, #11)
# --------------------------------------------------------------------------


def test_split_trials_by_speaker_is_speaker_disjoint() -> None:
    trials = _many_trials(n_genuine=20, n_impostor=20)
    calibration, evaluation, dropped = split_trials_by_speaker(trials, calibration_fraction=0.5, seed=42)

    assert calibration, "calibration split should not be empty for this fixture"
    assert evaluation, "evaluation split should not be empty for this fixture"
    overlap = speaker_overlap(calibration, evaluation)
    assert overlap == set()


def test_split_trials_by_speaker_accounts_for_every_trial() -> None:
    trials = _many_trials(n_genuine=10, n_impostor=10)
    calibration, evaluation, dropped = split_trials_by_speaker(trials, seed=1)
    assert len(calibration) + len(evaluation) + len(dropped) == len(trials)


def test_split_trials_by_speaker_deterministic() -> None:
    trials = _many_trials(n_genuine=10, n_impostor=10)
    cal_a, eval_a, drop_a = split_trials_by_speaker(trials, seed=99)
    cal_b, eval_b, drop_b = split_trials_by_speaker(trials, seed=99)
    assert [t.trial_id for t in cal_a] == [t.trial_id for t in cal_b]
    assert [t.trial_id for t in eval_a] == [t.trial_id for t in eval_b]


def test_speaker_overlap_detects_intentional_leakage() -> None:
    # Two trials that intentionally share a speaker across "calibration"
    # and "evaluation" groupings (simulating a caller who did NOT use
    # split_trials_by_speaker) -- speaker_overlap must catch this.
    calibration = [_make_trial("c1", "idShared", "idShared", label=1)]
    evaluation = [_make_trial("e1", "idShared", "idOther", label=0)]
    overlap = speaker_overlap(calibration, evaluation)
    assert overlap == {"idShared"}


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------


def test_summarize_trials_counts() -> None:
    trials = _many_trials(n_genuine=3, n_impostor=2)
    summary = summarize_trials(trials)
    assert summary.num_trials == 5
    assert summary.num_genuine == 3
    assert summary.num_impostor == 2
    # 3 genuine trials -> 3 unique speakers (same speaker both sides);
    # 2 impostor trials -> 4 unique speakers (idA0/idB0, idA1/idB1).
    assert summary.num_speakers == 3 + 4
    assert summary.num_utterances == 3 * 2 + 2 * 2
