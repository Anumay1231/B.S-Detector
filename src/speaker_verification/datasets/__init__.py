"""
speaker_verification.datasets
------------------------------

Dataset adapters for large-scale calibration (Phase 7).

- ``trials``: dataset-agnostic ``Trial`` representation plus sampling,
  speaker-disjoint calibration/evaluation splitting, and file-existence
  validation.
- ``voxceleb``: a VoxCeleb-specific adapter that parses VoxCeleb's
  official trial-list format into ``Trial`` objects. Future datasets
  should get their own sibling adapter module rather than adding
  dataset-specific branches to ``trials.py`` or the pipeline scripts.

Re-exported here for convenience:
"""

from __future__ import annotations

from speaker_verification.datasets.trials import (
    DatasetError,
    Trial,
    TrialSetSummary,
    TrialValidationError,
    sample_trials,
    speaker_overlap,
    split_trials_by_speaker,
    summarize_trials,
    validate_trials,
)
from speaker_verification.datasets.voxceleb import (
    TrialListParseError,
    VOXCELEB_SOURCE_URL,
    parse_trial_list,
    resolve_utterance_path,
    speaker_id_from_utterance_id,
)

__all__ = [
    "DatasetError",
    "Trial",
    "TrialSetSummary",
    "TrialValidationError",
    "sample_trials",
    "speaker_overlap",
    "split_trials_by_speaker",
    "summarize_trials",
    "validate_trials",
    "TrialListParseError",
    "VOXCELEB_SOURCE_URL",
    "parse_trial_list",
    "resolve_utterance_path",
    "speaker_id_from_utterance_id",
]
