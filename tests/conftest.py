"""
conftest.py
-----------

Makes the `speaker_verification` package (under src/) importable during
tests without requiring a separate install step (no pyproject.toml /
setup.py has been added yet — that is expected in a later phase). This is
a standard pytest pattern for "src layout" projects.

Also adds scripts/ to sys.path (Phase 7) so the Phase 7 pipeline
scripts' pure/testable functions (e.g. build_trials.build_subset,
extract_embeddings.extract_embeddings, score_trials.score_trials,
run_voxceleb_calibration.run_calibration) can be imported directly by
tests, the same way the package under src/ is.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
