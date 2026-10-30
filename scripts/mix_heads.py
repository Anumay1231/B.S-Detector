"""
Blend two SLS heads by weight averaging (WiSE-FT): (1 - alpha) * base + alpha * tuned.

    python scripts/mix_heads.py models/sls_head_silence_invariant.pth models/sls_head_phase12_full.pth \
        --alpha 0.3 --out models/sls_head_phase12.pth

The tuned head must be fine-tuned from the base head (same architecture, same
starting point). Phase 12: the fully retrained head reached 84% on held-out people
and voices but flagged clean LibriSpeech audiobook speech, a domain missing from
its training data; alpha 0.3 kept the base head's behaviour there (0 wrong in
scripts/ui_path_check.py) and still gained on held-out fakes and the phone line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def load(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k.removeprefix("module."): v for k, v in state.items() if not k.startswith("ssl_model.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("tuned", type=Path)
    ap.add_argument("--alpha", type=float, default=0.3, help="weight of the tuned head")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    base, tuned = load(a.base), load(a.tuned)
    if set(base) != set(tuned):
        raise SystemExit(f"the heads do not match: {sorted(set(base) ^ set(tuned))}")
    # integer buffers (BatchNorm's num_batches_tracked) come from the tuned head
    mixed = {k: (base[k].float() * (1 - a.alpha) + tuned[k].float() * a.alpha).to(base[k].dtype)
             if base[k].is_floating_point() else tuned[k] for k in base}
    torch.save(mixed, a.out)
    print(f"wrote {a.out} ({1 - a.alpha:.2f} x {a.base.name} + {a.alpha:.2f} x {a.tuned.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
