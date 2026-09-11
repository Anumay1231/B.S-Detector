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
for an impostor (different-speaker) trial.

Two utterance-path conventions are supported, because the published
VoxCeleb1 trial lists do not all use the same one:

1. **Canonical / numeric-id form** (matches the on-disk audio layout)::

       id10270/x6uYqmx31kE/00001.wav

2. **Speaker-name form** (used by the widely mirrored
   ``voxceleb1_test.txt`` / ``veri_test.txt`` list)::

       Eartha_Kitt/x6uYqmx31kE_0000001.wav

   Here the speaker is identified by their VGGFace1 *name* rather than
   their VoxCeleb1 id, and the video id and utterance number are joined
   by an underscore instead of a directory separator.

Form 2 cannot be resolved to a local file path on its own -- the
name -> id mapping is external data. It is published by VGG as
``vox1_meta.csv``, and this module reads that file rather than
guessing or hard-coding any mapping (see ``load_speaker_name_map`` and
``normalize_utterance_id``). Callers pass it via ``parse_trial_list(...,
meta_path=...)``; without it, a form-2 list raises a clear error
instead of silently mis-resolving paths.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from speaker_verification.datasets.trials import DatasetError, Trial

#: Official VoxCeleb dataset landing page. Actual audio download
#: requires VGG's own registration/access process; this project does
#: not bypass that (see module docstring).
VOXCELEB_SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/"

#: Filename of VGG's official VoxCeleb1 speaker-metadata table, which
#: maps VoxCeleb1 ids to VGGFace1 speaker names.
VOXCELEB_META_FILENAME = "vox1_meta.csv"

#: VoxCeleb speaker ids are of the form "idNNNNN" (e.g. "id10270"),
#: always the first path segment of a dataset-relative utterance path.
_SPEAKER_ID_PATTERN = re.compile(r"^(id\d+)/")

#: A bare VoxCeleb speaker id, with no trailing path separator.
_BARE_SPEAKER_ID_PATTERN = re.compile(r"^id\d+$")

#: Utterance numbers are zero-padded to this width on disk
#: ("00001.wav"), whereas speaker-name-form trial lists pad them to
#: seven digits ("..._0000001.wav"). Verified against the real
#: VoxCeleb1 test audio tree, whose utterance stems are all 5 digits.
_UTTERANCE_NUMBER_WIDTH = 5


class TrialListParseError(DatasetError):
    """Raised when a line in a VoxCeleb trial-list file cannot be parsed
    into (label, reference_path, test_path), or an utterance path is in
    a form this adapter cannot resolve to a local file."""


class SpeakerMetadataError(DatasetError):
    """Raised when VoxCeleb's speaker-metadata file (vox1_meta.csv)
    cannot be read or does not have the expected columns."""


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


def load_speaker_name_map(meta_path: str) -> Dict[str, str]:
    """Load VGG's ``vox1_meta.csv`` and return a mapping from VGGFace1
    speaker *name* to VoxCeleb1 *id*, e.g. ``{"Eartha_Kitt": "id10270"}``.

    The official file is tab-separated with a header row and these
    columns::

        VoxCeleb1 ID    VGGFace1 ID    Gender    Nationality    Set

    Only the first two columns are used here; gender/nationality/set are
    read but deliberately ignored, since this adapter's only job is path
    resolution. Whitespace around every field is stripped (some mirrors
    of this file pad the columns).

    Args:
        meta_path: Path to a local copy of ``vox1_meta.csv``.

    Returns:
        Mapping of speaker name -> VoxCeleb1 speaker id.

    Raises:
        SpeakerMetadataError: If the file does not exist, contains no
            usable rows, or maps one name to two different ids.
    """
    if not os.path.isfile(meta_path):
        raise SpeakerMetadataError(
            f"VoxCeleb speaker-metadata file not found: '{meta_path}'. "
            f"This file ({VOXCELEB_META_FILENAME}) is published alongside "
            f"the dataset ({VOXCELEB_SOURCE_URL}) and is required to "
            f"resolve speaker-name-form trial lists. It is not generated "
            f"or guessed by this project."
        )

    name_to_id: Dict[str, str] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Tab-separated officially; fall back to any whitespace for
            # mirrors that re-save it. Speaker names themselves never
            # contain whitespace (spaces are written as underscores).
            fields = [field.strip() for field in line.split("\t")]
            if len(fields) < 2:
                fields = [field.strip() for field in line.split()]
            if len(fields) < 2:
                continue

            speaker_id, speaker_name = fields[0], fields[1]
            if not _BARE_SPEAKER_ID_PATTERN.match(speaker_id):
                # Header row ("VoxCeleb1 ID") or a comment -- skip it
                # rather than treating it as data.
                continue

            existing = name_to_id.get(speaker_name)
            if existing is not None and existing != speaker_id:
                raise SpeakerMetadataError(
                    f"{meta_path}:{line_num}: speaker name "
                    f"{speaker_name!r} maps to both {existing!r} and "
                    f"{speaker_id!r}. Cannot resolve utterance paths "
                    f"unambiguously."
                )
            name_to_id[speaker_name] = speaker_id

    if not name_to_id:
        raise SpeakerMetadataError(
            f"No usable speaker rows found in '{meta_path}'. Expected "
            f"tab-separated lines whose first column is a VoxCeleb id "
            f"('idNNNNN') and whose second column is the speaker name."
        )

    return name_to_id


def normalize_utterance_id(
    utterance_id: str, name_to_id: Optional[Dict[str, str]] = None
) -> str:
    """Normalize a trial-list utterance id to the canonical, on-disk
    form ``<speaker_id>/<video_id>/<utterance>.wav``.

    Canonical ids are returned unchanged. Speaker-name-form ids
    (``Eartha_Kitt/x6uYqmx31kE_0000001.wav``) are converted using
    ``name_to_id``::

        Eartha_Kitt/x6uYqmx31kE_0000001.wav
            -> id10270/x6uYqmx31kE/00001.wav

    The utterance number is re-padded rather than truncated (7-digit
    ``0000001`` -> 5-digit ``00001``), and the split is on the *last*
    underscore, because VoxCeleb video ids can themselves contain
    underscores (e.g. ``5sJomL_D0_g``).

    Args:
        utterance_id: The path as written in the trial list.
        name_to_id: Speaker-name -> speaker-id mapping from
            ``load_speaker_name_map``. Only needed for name-form ids.

    Returns:
        The canonical dataset-relative utterance id.

    Raises:
        TrialListParseError: If the id is in neither supported form, or
            is name-form but ``name_to_id`` is missing or lacks the name.
    """
    normalized = utterance_id.replace("\\", "/").strip()
    segments = normalized.split("/")

    # Form 1: already canonical (id10270/x6uYqmx31kE/00001.wav).
    if len(segments) == 3 and _BARE_SPEAKER_ID_PATTERN.match(segments[0]):
        return normalized

    # Form 2: speaker-name form (Eartha_Kitt/x6uYqmx31kE_0000001.wav).
    if len(segments) == 2:
        speaker_name, filename = segments
        if name_to_id is None:
            raise TrialListParseError(
                f"Utterance id {utterance_id!r} uses the speaker-name "
                f"form ('<Name>/<video_id>_<number>.wav'), which cannot "
                f"be resolved without VoxCeleb's speaker-metadata file "
                f"({VOXCELEB_META_FILENAME}). Pass it via "
                f"--meta / meta_path so the name can be mapped to its "
                f"'idNNNNN' speaker id."
            )

        speaker_id = name_to_id.get(speaker_name)
        if speaker_id is None:
            raise TrialListParseError(
                f"Speaker name {speaker_name!r} (from utterance id "
                f"{utterance_id!r}) is not present in the supplied "
                f"VoxCeleb speaker metadata. Check that the metadata "
                f"file matches this trial list's dataset version -- "
                f"this project does not guess speaker identities."
            )

        stem, extension = os.path.splitext(filename)
        if "_" not in stem:
            raise TrialListParseError(
                f"Cannot split utterance id {utterance_id!r} into "
                f"'<video_id>_<number>{extension or '.wav'}': no "
                f"underscore found in {filename!r}."
            )

        video_id, number = stem.rsplit("_", 1)
        if not number.isdigit() or not video_id:
            raise TrialListParseError(
                f"Cannot parse utterance number from {utterance_id!r}: "
                f"expected '<video_id>_<digits>{extension or '.wav'}', "
                f"got {filename!r}."
            )

        padded = f"{int(number):0{_UTTERANCE_NUMBER_WIDTH}d}"
        return f"{speaker_id}/{video_id}/{padded}{extension}"

    raise TrialListParseError(
        f"Unrecognized VoxCeleb utterance id {utterance_id!r}. Expected "
        f"either 'idNNNNN/<video_id>/<utterance>.wav' or "
        f"'<SpeakerName>/<video_id>_<number>.wav'."
    )


def resolve_utterance_path(audio_root: str, utterance_id: str) -> str:
    """Resolve a dataset-relative utterance id to a local filesystem
    path under ``audio_root``. Does not check that the file exists --
    see trials.validate_trials() for that."""
    normalized = utterance_id.replace("\\", "/")
    return os.path.join(audio_root, *normalized.split("/"))


# --------------------------------------------------------------------------
# Trial-list parsing
# --------------------------------------------------------------------------


def parse_trial_list(
    trial_list_path: str, audio_root: str, meta_path: Optional[str] = None
) -> List[Trial]:
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
            organized as ``<audio_root>/<speaker_id>/<video_id>/<utt>.wav``.
        meta_path: Optional path to VGG's ``vox1_meta.csv``. Required
            only when the trial list uses the speaker-name form (see the
            module docstring); ignored for canonical numeric-id lists.

    Returns:
        A list of ``Trial`` objects, in file order, each with a
        ``trial_id`` derived from the trial-list filename and line
        number (so it is stable and traceable back to its source line).
        Utterance ids are normalized to the canonical
        ``idNNNNN/<video_id>/<utt>.wav`` form regardless of which input
        form the trial list used, so downstream stages (embedding cache
        keys, scoring, reports) see one consistent identifier.

    Raises:
        DatasetError: If ``trial_list_path`` does not exist.
        SpeakerMetadataError: If ``meta_path`` is given but unusable.
        TrialListParseError: If a line is malformed (wrong number of
            fields, a label that isn't "0"/"1", or an utterance id that
            cannot be resolved to the canonical form).
    """
    if not os.path.isfile(trial_list_path):
        raise DatasetError(
            f"VoxCeleb trial-list file not found: '{trial_list_path}'. "
            f"This file is not generated automatically -- it must be "
            f"obtained from the official VoxCeleb site "
            f"({VOXCELEB_SOURCE_URL}). See docs/calibration.md."
        )

    name_to_id = load_speaker_name_map(meta_path) if meta_path else None

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

            try:
                reference_id = normalize_utterance_id(reference_id, name_to_id)
                test_id = normalize_utterance_id(test_id, name_to_id)
            except TrialListParseError as exc:
                raise TrialListParseError(
                    f"{trial_list_path}:{line_num}: {exc}"
                ) from exc

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
