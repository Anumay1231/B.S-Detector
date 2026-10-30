"""
Held-out numbers for every detector head (handover task 14).

    python scripts/eval_phase12.py [--heads models/a.pth models/b.pth] [--per-source 200]

Scores each head on the phase 12 *test* split: people and TTS voices that no
head's training or checkpoint choice saw (phase12_common.py), with demo speakers
excluded. Every clip is heard under one of the conditions in train_phase12.py
(start, phone line, MP3, Opus, mid-utterance, added quiet, noise, room, replay),
cycling through them. Reports, per head, the balanced accuracy at the app's 0.5
cut-off overall, per condition, and the share of each source called correctly.

Writes outputs/phase12_eval.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase12_common import ROOT  # noqa: E402
from train_phase12 import CONDITIONS, collect, fixed_set, load_models, report, score  # noqa: E402

HEADS = [
    ROOT / "models" / "sls_head_silence_invariant.pth",
    ROOT / "models" / "sls_head_n500_indicsynth_best.pth",
    ROOT / "models" / "sls_head_channel_robust.pth",
    ROOT / "models" / "sls_head_phase12.pth",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=Path, nargs="*", default=HEADS)
    ap.add_argument("--per-source", type=int, default=200)
    a = ap.parse_args()
    heads = [h for h in a.heads if h.is_file()]

    items = collect()
    test = fixed_set([it for it in items if it["split"] == "test"], seed=2, per_source=a.per_source)
    del items
    waves = [w for _, _, w in test]
    n_real = sum(it["label"] == 1 for it, _, _ in test)
    print(f"test windows: {len(test)} ({n_real} real, {len(test) - n_real} fake)", flush=True)

    backbone, head = load_models("cuda", heads[0])
    results = {}
    for h in heads:
        state = torch.load(h, map_location="cpu", weights_only=True)
        head.load_state_dict({k.removeprefix("module."): v for k, v in state.items() if not k.startswith("ssl_model.")})
        results[h.name] = report(test, score(backbone, head, waves, "cuda"))
        r = results[h.name]
        print(f"{h.name}: BA {r['balancedAccuracy']:.3f}, worst condition {r['worstCondition']:.3f}", flush=True)

    out = ROOT / "outputs" / "phase12_eval.json"
    out.write_text(json.dumps({"windows": len(test), "real": n_real, "results": results}, indent=1), encoding="utf-8", newline="")

    names = list(results)
    print("\n| Condition | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    print("| **All** | " + " | ".join(f"**{100 * results[n]['balancedAccuracy']:.1f}%**" for n in names) + " |")
    for c in CONDITIONS:
        print(f"| {c} | " + " | ".join(f"{100 * results[n]['byCondition'][c]:.1f}%" for n in names) + " |")
    print("\n| Source (share correct) | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for s in sorted(results[names[0]]["bySource"]):
        print(f"| {s} | " + " | ".join(f"{100 * results[n]['bySource'][s]:.1f}%" for n in names) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
