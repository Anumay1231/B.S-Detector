"""
trials.py
---------

Dataset-agnostic trial representation and operations.

This module knows nothing about VoxCeleb, file formats, or audio. It
operates purely on the standardized ``Trial`` representation: a
reference/test utterance pair, a genuine (1) / impostor (0) label, and
the speaker identities of both utterances. Format-specific parsing
(reading a particular dataset's trial-list file and resolving paths on
disk) lives in sibling adapter modules such as ``voxceleb.py``, which
produce ``Trial`` objects for this module to operate on. This split
exists so a future second dataset adapter would not need to duplicate
sampling/splitting/validation logic.

Functionality here:
    - ``Trial``: the standardized trial representation.
    - ``validate_trials()``: separates trials into those whose audio
      files exist on disk vs. those that are missing (existence check
      only -- decoding is audio.py's job, at embedding-extraction time).
    - ``sample_trials()``: deterministic, seeded subset sampling that
      avoids "first N" ordering bias.
    - ``split_trials_by_speaker()``: a calibration/evaluation split with
      zero speaker overlap by construction.
    - ``speaker_overlap()``: independently re-checks that a
      calibration/evaluation split is actually speaker-disjoint (rather
      than only trusting split_trials_by_speaker's construction).
    - ``summarize_trials()``: descriptive counts (speakers, utterances,
      genuine/impostor trials) used for Phase 7's reporting requirements.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple


class DatasetError(Exception):
    """Base class for all errors raised by the datasets package."""


class TrialValidationError(DatasetError):
    """Raised when trial data is structurally invalid (e.g. a label that
    is neither 0 nor 1)."""


# --------------------------------------------------------------------------
# Trial
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One standardized genuine/impostor trial pair.

    Attributes:
        trial_id: Stable identifier for this trial (e.g. derived from
            the source trial-list filename and line number).
        reference_id: Dataset-relative utterance identifier for the
            reference file (e.g. "id10270/x6uYqmx31kE/00001.wav" for
            VoxCeleb). This is the embedding-cache key -- two trials
            that reference the same utterance share the same
            reference_id/test_id and therefore the same cached
            embedding.
        test_id: Same convention as reference_id, for the test file.
        reference_path: Resolved local filesystem path to the reference
            audio file.
        test_path: Resolved local filesystem path to the test audio
            file.
        label: 1 = genuine (same speaker), 0 = impostor (different
            speaker).
        reference_speaker: Speaker identifier for the reference file.
        test_speaker: Speaker identifier for the test file.
    """

    trial_id: str
    reference_id: str
    test_id: str
    reference_path: str
    test_path: str
    label: int
    reference_speaker: str
    test_speaker: str

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise TrialValidationError(
                f"Trial {self.trial_id!r} has invalid label "
                f"{self.label!r}; must be 1 (genuine) or 0 (impostor)."
            )
        if self.label == 1 and self.reference_speaker != self.test_speaker:
            raise TrialValidationError(
                f"Trial {self.trial_id!r} is labeled genuine (1) but its "
                f"reference speaker ({self.reference_speaker!r}) and test "
                f"speaker ({self.test_speaker!r}) differ. A genuine trial "
                f"must be a same-speaker pair."
            )
        if self.label == 0 and self.reference_speaker == self.test_speaker:
            raise TrialValidationError(
                f"Trial {self.trial_id!r} is labeled impostor (0) but its "
                f"reference and test speaker are both "
                f"{self.reference_speaker!r}. An impostor trial must be a "
                f"different-speaker pair."
            )

    @property
    def is_genuine(self) -> bool:
        return self.label == 1


# --------------------------------------------------------------------------
# File-existence validation (NOT audio decoding -- see audio.py)
# --------------------------------------------------------------------------


def validate_trials(trials: Sequence[Trial]) -> Tuple[List[Trial], List[Trial]]:
    """Split trials into (valid, missing) based on whether BOTH the
    reference and test audio files exist on disk.

    This is a cheap existence check only -- it does not open, decode,
    or otherwise validate the audio content (that happens later, per
    file, in audio.py/encoder.py during embedding extraction, which
    handles corrupt/undecodable files with its own clearly-named
    exceptions).
    """
    import os

    valid: List[Trial] = []
    missing: List[Trial] = []
    for trial in trials:
        if os.path.isfile(trial.reference_path) and os.path.isfile(trial.test_path):
            valid.append(trial)
        else:
            missing.append(trial)
    return valid, missing


# --------------------------------------------------------------------------
# Deterministic subset sampling
# --------------------------------------------------------------------------


def sample_trials(
    trials: Sequence[Trial],
    max_genuine: "int | None" = None,
    max_impostor: "int | None" = None,
    seed: int = 42,
) -> List[Trial]:
    """Deterministically sample a bounded subset of trials.

    Sampling strategy (documented per Phase 7B -- NOT "first N"):
        Trial-list files (including VoxCeleb's official protocol files)
        are commonly ordered by speaker block, i.e. all trials
        involving one reference speaker/utterance appear consecutively.
        Taking the first N trials would therefore concentrate the
        subset on a handful of early speakers/utterances and bias
        anything calibrated from it. Instead:

        1. Genuine and impostor trials are separated (they are bounded
           independently by max_genuine/max_impostor).
        2. Each list is shuffled with a seeded ``random.Random(seed)``
           instance -- deterministic for a given (trials, seed) pair,
           but decoupled from the original file order.
        3. The first ``max_genuine`` / ``max_impostor`` entries of each
           shuffled list are kept (None means "keep all").

    This is a uniform random sample without replacement from whatever
    trials were provided. It does not by itself guarantee balanced
    speaker coverage beyond what's already present in the input list --
    see ``split_trials_by_speaker`` for the speaker-disjointness
    guarantee used downstream for calibration/evaluation.

    Determinism: calling this twice with the same `trials` (same
    objects, same order) and the same `seed` always returns the same
    subset, in the same order.
    """
    genuine = [t for t in trials if t.is_genuine]
    impostor = [t for t in trials if not t.is_genuine]

    rng = random.Random(seed)
    genuine_shuffled = list(genuine)
    rng.shuffle(genuine_shuffled)
    impostor_shuffled = list(impostor)
    rng.shuffle(impostor_shuffled)

    if max_genuine is not None:
        genuine_shuffled = genuine_shuffled[:max_genuine]
    if max_impostor is not None:
        impostor_shuffled = impostor_shuffled[:max_impostor]

    return genuine_shuffled + impostor_shuffled


# --------------------------------------------------------------------------
# Calibration / evaluation split (speaker-disjoint by construction)
# --------------------------------------------------------------------------


def split_trials_by_speaker(
    trials: Sequence[Trial],
    calibration_fraction: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Trial], List[Trial], List[Trial]]:
    """Split trials into (calibration, evaluation, dropped) sets with
    ZERO speaker overlap between calibration and evaluation, by
    construction.

    Method:
        1. Collect every unique speaker id appearing as either the
           reference or test speaker across `trials`.
        2. Deterministically shuffle that speaker list with
           ``random.Random(seed)`` and assign the first
           ``round(calibration_fraction * num_speakers)`` speakers to
           CALIBRATION, the remainder to EVALUATION.
        3. A trial is kept in the calibration set only if BOTH its
           reference and test speaker are on the calibration side; kept
           in the evaluation set only if BOTH are on the evaluation
           side. Trials whose two speakers fall on opposite sides are
           DROPPED -- keeping them in either set would leak a speaker
           across the calibration/evaluation boundary. Callers should
           report ``len(dropped)`` (see scripts/run_voxceleb_calibration.py).

    This makes the split speaker-disjoint by construction; ``speaker_overlap()``
    independently re-verifies that rather than only trusting this
    logic.

    Note on official VoxCeleb protocols: the official VoxCeleb1 /
    VoxCeleb1-E / VoxCeleb1-H test trial lists are themselves NOT split
    into "calibration" and "evaluation" halves -- they are a single
    evaluation protocol. This function applies an additional,
    project-specific speaker-level split on top of whatever trial list
    is supplied, specifically so a subset of it can be used to choose a
    threshold (calibration) while a disjoint subset is held out to
    measure that threshold's performance (evaluation) -- see
    docs/calibration.md, "Phase 7", for the full discussion.
    """
    speakers = sorted(
        {t.reference_speaker for t in trials} | {t.test_speaker for t in trials}
    )
    rng = random.Random(seed)
    shuffled = list(speakers)
    rng.shuffle(shuffled)
    n_calibration = round(calibration_fraction * len(shuffled))
    calibration_speakers = set(shuffled[:n_calibration])
    evaluation_speakers = set(shuffled[n_calibration:])

    calibration: List[Trial] = []
    evaluation: List[Trial] = []
    dropped: List[Trial] = []
    for trial in trials:
        in_calibration = (
            trial.reference_speaker in calibration_speakers
            and trial.test_speaker in calibration_speakers
        )
        in_evaluation = (
            trial.reference_speaker in evaluation_speakers
            and trial.test_speaker in evaluation_speakers
        )
        if in_calibration:
            calibration.append(trial)
        elif in_evaluation:
            evaluation.append(trial)
        else:
            dropped.append(trial)

    return calibration, evaluation, dropped


def speaker_overlap(calibration: Sequence[Trial], evaluation: Sequence[Trial]) -> Set[str]:
    """Return the set of speaker ids present in BOTH the calibration and
    evaluation trial sets. An empty set means the split is
    speaker-disjoint. This is an independent re-check -- it does not
    assume split_trials_by_speaker() was used or was correct.
    """
    calibration_speakers = {t.reference_speaker for t in calibration} | {
        t.test_speaker for t in calibration
    }
    evaluation_speakers = {t.reference_speaker for t in evaluation} | {
        t.test_speaker for t in evaluation
    }
    return calibration_speakers & evaluation_speakers


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialSetSummary:
    """Descriptive counts for a set of trials (Phase 7G reporting)."""

    num_trials: int
    num_genuine: int
    num_impostor: int
    num_speakers: int
    num_utterances: int


def summarize_trials(trials: Sequence[Trial]) -> TrialSetSummary:
    """Compute descriptive counts: trials, genuine/impostor breakdown,
    unique speakers, and unique utterances (by reference_id/test_id, so
    an utterance referenced by multiple trials is counted once)."""
    num_genuine = sum(1 for t in trials if t.is_genuine)
    speakers = {t.reference_speaker for t in trials} | {t.test_speaker for t in trials}
    utterances = {t.reference_id for t in trials} | {t.test_id for t in trials}
    return TrialSetSummary(
        num_trials=len(trials),
        num_genuine=num_genuine,
        num_impostor=len(trials) - num_genuine,
        num_speakers=len(speakers),
        num_utterances=len(utterances),
    )
