"""
Original vs Indian-adapted ECAPA on a telephone line, on people the adapted model never heard.

    python scripts/sv_duration.py embed --model ecapa --also ecapa_adapted wespeaker_r34 \
        --only svq_hi svq_en cv_hi_test --conds phone phone_room0 --suffix _phone
    python scripts/sv_phone_fair.py

For each ECAPA, the fused decision constants are fitted the fair way (sv_fusion.py
--fair: Common Voice train halves + SVQ fold 0, clean and room audio), then pairs are
scored on SVQ fold 1 and Common Voice Hindi part 1 after the phone line of
scripts/ui_path_check.py (8 kHz band, 8-bit mu-law), alone or after a simulated room.
Reports the decision the app would give: same-person pairs accepted, different-person
pairs wrongly accepted (the dangerous error), "not sure", and accuracy on answers.
Writes outputs/sv_bench/phone_fair.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sv_benchmark as sb  # noqa: E402
import sv_duration as sd  # noqa: E402
import sv_fusion as sf  # noqa: E402

SETS = (("svq_hi_phone", 1), ("svq_en_phone", 1), ("cv_hi_test_phone", None))
BUCKETS = ((3, 8, "3-8 s"), (8, 99, "8 s+"))


def main() -> int:
    out: dict = {}
    for ecapa in ("ecapa", "ecapa_adapted"):
        sf.ECAPA_MODEL = ecapa
        k = sf.fit_constants(sf.merge([sf.load_set(n) for n in sd.FIT_SETS] + [sf.load_set(n, 0) for n in sd.TEST_SETS]))
        out[ecapa] = {}
        for name, fold in SETS:
            s = sf.load_set(name, fold)
            score, thr, mfn = sf.score_with(k, "fused", s)
            t, m = thr(s["dmin"]), mfn(s["dmin"])
            match, nonmatch = score >= t + m, score < t - m
            for cond in np.unique(s["cond"]):
                for lo, hi, label in BUCKETS:
                    sel = (s["cond"] == cond) & (s["dmin"] >= lo) & (s["dmin"] < hi)
                    g, i = sel & (s["y"] == 1), sel & (s["y"] == 0)
                    if g.sum() < 10 or i.sum() < 10:
                        continue
                    answered_g, answered_i = g & (match | nonmatch), i & (match | nonmatch)
                    acc = 0.5 * (np.mean(match[answered_g]) + np.mean(nonmatch[answered_i])) if answered_g.any() and answered_i.any() else float("nan")
                    out[ecapa][f"{name}/{cond}/{label}"] = {
                        "genuine": int(g.sum()), "impostor": int(i.sum()),
                        "genuineAccepted": float(np.mean(match[g])),
                        "impostorAccepted": float(np.mean(match[i])),
                        "unsure": float(np.mean(~(match | nonmatch)[sel])),
                        "accuracyOnAnswers": float(acc),
                    }
    (sb.OUT / "phone_fair.json").write_text(json.dumps(out, indent=1))

    a, b = out["ecapa"], out["ecapa_adapted"]
    print(f"{'cell':38s} {'pairs':>10s} | {'impostors accepted':^19s} | {'accuracy on answers':^21s} | {'not sure':^15s}")
    print(f"{'':38s} {'gen/imp':>10s} | {'orig':>8s} {'adapt':>8s}  | {'orig':>9s} {'adapt':>9s}  | {'orig':>6s} {'adapt':>6s}")
    for cell in a:
        x, y = a[cell], b.get(cell)
        if not y:
            continue
        print(f"{cell:38s} {x['genuine']:>4d}/{x['impostor']:<5d} | {100 * x['impostorAccepted']:7.2f}% {100 * y['impostorAccepted']:7.2f}%  |"
              f" {100 * x['accuracyOnAnswers']:8.2f}% {100 * y['accuracyOnAnswers']:8.2f}%  | {100 * x['unsure']:5.1f}% {100 * y['unsure']:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
