"""
voxceleb.py
-----------

VoxCeleb-specific dataset adapter: turns an official VoxCeleb
verification trial-list file (plus a local directory of already-obtained
VoxCeleb audio) into the dataset-agnostic ``Trial`` representation
defined in ``trials.py``. All sampling, splitting, and validation logic
that does NOT depend on VoxCeleb's specific file layout lives in
``trials.py`` -- this module is deliberately kept to parsing and path
resolution only, per Phase 7A ("create a dataset adapter rather than
hard-coding VoxCeleb assumptions throughout the project").

Official dataset source (see docs/calibration.md, "Phase 7", for the
full write-up):

    https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

The official site describes VoxCeleb1 (1,251 speakers, 150,000+
utterances) and VoxCeleb2 (6,112 speakers, 1,000,000+ utterances), and
publishes established verification trial lists (VoxCeleb1, VoxCeleb1-E,
VoxCeleb1-H, and "cleaned" variants of each). Downloading the actual
audio requires completing VGG's own registration/access process --
THIS MODULE DOES NOT DOWNLOAD, SCRAPE, OR BYPASS THAT PROCESS. It only
parses a trial-list file and audio directory tree that the caller has
already obtained legitimately and placed on disk (see
docs/calibration.md for exactly what to obtain and where to put it).

Trial-list format (VoxCeleb1 / VoxCeleb1-E / VoxCeleb1-H test lists, as
published by VGG): one trial per line, whitespace-separated:

    <label> <reference_relative_path> <test_relative_path>

where ``label`` is ``1`` for a genuine (same-speaker) trial and ``0``
for an impostor (different-speaker) trial, and each relative path has
the form ``<speaker_id>/<video_id>/<utterance_file>``, e.g.
``id10270/x6uYqmx31kE/00001.wav``.
"""

from __future__ import annotations

import os
import re
from typing import List

from speaker_verification.datasets.trials import DatasetError, Trial

#: Official VoxCeleb dataset landing page. Actual audio download
#: requires VGG's own registration/access process; this project does
#: not bypass that (see module docstring).
VOXCELEB_SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/"

#: VoxCeleb speaker ids are of the form "idNNNNN" (e.g. "id10270"),
#: always the first path segment of a dataset-relative utterance path.
_SPEAKER_ID_PATTERN = re.compile(r"^(id\d+)/")


class TrialListParseError(DatasetError):
    """Raised when a line in a VoxCeleb trial-list file cannot be parsed
    into (label, reference_path, test_path), or a path is missing the
    expected VoxCeleb speaker-id prefix."""


# --------------------------------------------------------------------------
# Path / speaker-id helpers
# --------------------------------------------------------------------------


def speaker_id_from_utterance_id(utterance_id: str) -> str:
    """Extract the VoxCeleb speaker id (e.g. "id10270") from a
    dataset-relative utterance path such as
    "id10270/x6uYqmx31kE/00001.wav".

    Raises:
        TrialListParseError: If ``utterance_id`` does not start with
            the expected "idNNNNN/" speaker-id prefix.
    """
    normalized = utterance_id.replace("\\", "/")
    match = _SPEAKER_ID_PATTERN.match(normalized)
    if not match:
        raise TrialListParseError(
            f"Could not extract a VoxCeleb speaker id from utterance id "
            f"{utterance_id!r}; expected it to start with an 'idNNNNN/' "
            f"prefix (e.g. 'id10270/x6uYqmx31kE/00001.wav')."
        )
    return match.group(1)


def resolve_utterance_path(audio_root: str, utterance_id: str) -> str:
    """Resolve a dataset-relative utterance id to a local filesystem
    path under ``audio_root``. Does not check that the file exists --
    see trials.validate_trials() for that."""
    normalized = utterance_id.replace("\\", "/")
    return os.path.join(audio_root, *normalized.split("/"))


# --------------------------------------------------------------------------
# Trial-list parsing
# --------------------------------------------------------------------------


def parse_trial_list(trial_list_path: str, audio_root: str) -> List[Trial]:
    """Parse an official-format VoxCeleb trial-list file into ``Trial``
    objects, resolving each relative utterance path against
    ``audio_root``.

    This function does NOT check whether the resolved audio files
    actually exist (see ``trials.validate_trials()``) and does NOT
    load, decode, or embed any audio -- it is pure text parsing plus
    path arithmetic, deliberately kept separate from model inference
    (Phase 7A requirement).

    Args:
        trial_list_path: Path to a VoxCeleb-format trial-list text file
            (one "<label> <reference> <test>" line per trial; blank
            lines are skipped).
        audio_root: Local directory containing the VoxCeleb audio,
            organized as ``<audio_root>/<speaker_id>/<video_id>/<utt>.wav``
            -- the same relative layout the trial-list paths use.

    Returns:
        A list of ``Trial`` objects, in file order, each with a
        ``trial_id`` derived from the trial-list filename and line
        number (so it is stable and traceable back to its source line).

    Raises:
        DatasetError: If ``trial_list_path`` does not exist.
        TrialListParseError: If a line is malformed (wrong number of
            fields, a label that isn't "0"/"1", or a path missing the
            expected VoxCeleb speaker-id prefix).
    """
    if not os.path.isfile(trial_list_path):
        raise DatasetError(
            f"VoxCeleb trial-list file not found: '{trial_list_path}'. "
            f"This file is not generated automatically -- it must be "
            f"obtained from the official VoxCeleb site "
            f"({VOXCELEB_SOURCE_URL}). See docs/calibration.md."
        )

    trials: List[Trial] = []
    list_name = os.path.basename(trial_list_path)
    with open(trial_list_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                raise TrialListParseError(
                    f"{trial_list_path}:{line_num}: expected 3 "
                    f"whitespace-separated fields (label, reference_path, "
                    f"test_path), got {len(parts)}: {line!r}"
                )
            label_str, reference_id, test_id = parts
            if label_str not in ("0", "1"):
                raise TrialListParseError(
                    f"{trial_list_path}:{line_num}: label must be '0' "
                    f"(impostor) or '1' (genuine), got {label_str!r}."
                )
            label = int(label_str)

            reference_speaker = speaker_id_from_utterance_id(reference_id)
            test_speaker = speaker_id_from_utterance_id(test_id)

            trials.append(
                Trial(
                    trial_id=f"{list_name}:{line_num}",
                    reference_id=reference_id,
                    test_id=test_id,
                    reference_path=resolve_utterance_path(audio_root, reference_id),
                    test_path=resolve_utterance_path(audio_root, test_id),
                    label=label,
                    reference_speaker=reference_speaker,
                    test_speaker=test_speaker,
                )
            )

    return trials
