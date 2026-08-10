"""
conftest.py
-----------

Makes the `speaker_verification` package (under src/) importable during
tests without requiring a separate install step (no pyproject.toml /
setup.py has been added yet — that is expected in a later phase). This is
a standard pytest pattern for "src layout" projects.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
