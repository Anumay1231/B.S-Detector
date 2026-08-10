# Calibration and the verification decision (Phase 6)

This document explains what "calibration" means in this project, why a
raw cosine similarity score is not automatically a verification
decision, and exactly what data this project currently has (and lacks)
to produce one.

## Why a raw score is not a decision

`similarity.cosine_similarity()` (Phase 5) returns a number in roughly
`[-1.0, 1.0]`. That number alone does not tell you whether two
recordings are from the same speaker — it only becomes meaningful when
compared against a **threshold**: scores at or above the threshold are
called `MATCH`, scores below are called `NON_MATCH`.

The threshold is not a constant of nature. It depends on the specific
encoder model, the audio conditions, and — critically — where you choose
to trade off two kinds of error:

- **False Acceptance Rate (FAR):** the fraction of *impostor* trials
  (different speakers) that a given threshold would incorrectly accept
  as a `MATCH`. `FAR(t) = P(impostor_score >= t)`.
- **False Rejection Rate (FRR):** the fraction of *genuine* trials (same
  speaker) that a given threshold would incorrectly reject as a
  `NON_MATCH`. `FRR(t) = P(genuine_score < t)`.

As the threshold rises, FAR falls and FRR rises (stricter matching:
fewer false accepts, more false rejects), and vice versa. There is no
single "correct" threshold in the abstract — only a threshold that is
appropriate given a chosen operating point on this trade-off, estimated
from real labeled data.

## Equal Error Rate (EER)

This project's calibration module (`src/speaker_verification/calibration.py`)
selects its threshold using the **Equal Error Rate (EER)** operating
point: the threshold at which FAR and FRR are equal (or as close to
equal as the data allows). This is a standard, well-documented default
for biometric/speaker verification systems — not the only valid choice,
but a reasonable and explainable one. `CalibrationResult.eer` is the
error rate at that point (both FAR and FRR, approximately);
`CalibrationResult.eer_threshold` is the threshold that achieves it.

**Method:** FAR(t) and FRR(t) are evaluated at every candidate threshold
(the midpoint between each pair of consecutive unique observed scores,
plus sentinels below the minimum and above the maximum score, so the
curve's endpoints are always represented). As the threshold increases,
`FAR - FRR` is provably non-increasing. We scan for where it crosses
zero:

- If some candidate has `FAR == FRR` exactly, that is the EER point.
- Otherwise, we linearly interpolate between the last candidate with
  `FAR >= FRR` and the next candidate with `FAR < FRR` to estimate the
  crossing threshold and the FAR/FRR value there.
- If the curves never cross (e.g. perfect separation, where there is a
  whole gap of thresholds with `FAR = FRR = 0`), the candidate
  minimizing `|FAR - FRR|` is used.

`compute_eer()` documents this in its docstring and returns the method
description alongside the numbers so it's always traceable.

**Why not a fixed threshold like 0.5, 0.6, 0.7, or 0.75?** Because none
of those numbers are derived from this project's actual score
distributions — they would be guesses. `calibration.py` will not select
(and `verifier.SpeakerVerifier` will not silently default to) any such
value. A threshold is only produced by `calibrate()` from real labeled
genuine/impostor scores, or must be supplied explicitly by the caller
(and, if not from real calibration, explicitly marked
`is_calibrated=False`).

## Threshold direction

Higher cosine similarity means more likely same-speaker. The decision
rule, used consistently by `verifier.SpeakerVerifier`, is:

```
score >= threshold  ->  MATCH
score <  threshold  ->  NON_MATCH
```

A score exactly equal to the threshold is a `MATCH` (the boundary
belongs to the accept side).

## Data requirement for real calibration

A meaningful threshold requires **both**:

- A set of **genuine trials**: cosine similarity scores between two
  recordings of the *same* speaker.
- A set of **impostor trials**: cosine similarity scores between
  recordings of *different* speakers.

`calibrate()` raises `InsufficientCalibrationDataError` if either set is
empty — calibration is explicitly reported as **unavailable** in that
case, never silently defaulted.

Even with nonzero data, small samples are not statistically reliable.
This module uses `MIN_RELIABLE_TRIALS_PER_CLASS = 30` as a documented
(not rigorously derived — a commonly cited rule of thumb) minimum per
class; below that, `CalibrationResult.is_statistically_reliable` is
`False` and `CalibrationResult.warnings` explains why. The math is still
computed exactly for whatever data is given — the flag communicates
trustworthiness, not correctness of arithmetic.

## Current project status: calibration is NOT available

As of Phase 6, this project has:

- **Two genuine recordings** from one speaker (used in Phase 5 to obtain
  a real same-speaker cosine similarity score).
- **Zero impostor recordings** (no second speaker's audio exists in this
  project yet).

This means:

- We **can** demonstrate a single genuine trial.
- We **cannot** compute a meaningful impostor distribution (zero data
  points).
- We **cannot** compute a valid EER (requires both classes).
- We **cannot** claim a production threshold.
- We **cannot** claim any verification accuracy number.

Calling `calibrate([0.635386], [])` (our one real genuine score, zero
impostor scores) raises `InsufficientCalibrationDataError` — this is
verified directly by
`tests/test_calibration.py::test_only_genuine_scores_raises`.

### Our one real score: an uncalibrated example only

On the developer's machine (NVIDIA GeForce RTX 3060 Laptop GPU, CUDA
12.4), two genuine recordings of the same speaker were run through the
real pipeline (`preprocess_audio` → `SpeakerEncoder` → `cosine_similarity`)
and produced:

> **Uncalibrated genuine-trial example: cosine similarity = 0.635386**
> (192-dim embeddings, cosine similarity computation itself ~53.43 ms)

This number is real (not fabricated) and is recorded here purely as a
reference example of what the real pipeline produces on real audio. It
is explicitly **not**:

- a threshold,
- evidence that "0.635 means same speaker" in general,
- a statement about this system's accuracy,
- used anywhere in the code as a default/hard-coded value.

It is exactly one genuine data point out of the (at minimum) dozens per
class that would be needed for a statistically reliable calibration.

## How to actually calibrate this system

1. Collect multiple recordings from multiple speakers (ideally ≥30
   genuine pairs and ≥30 impostor pairs, per
   `MIN_RELIABLE_TRIALS_PER_CLASS`, and representative of the real
   deployment conditions — including Hindi/Hinglish speech, since this
   project's target use case has not been validated on that data at
   all; see docs/model.md).
2. Build `data/trials.csv` with columns `reference,test,label` (`label`
   is `genuine` or `impostor`) — see `scripts/calibrate.py --help` for
   the exact format and an inline example.
3. Run `python scripts/calibrate.py --trials data/trials.csv`. It scores
   every pair with the real encoder + `cosine_similarity`, then calls
   `calibration.calibrate()` and prints the full FAR/FRR/EER report,
   including the reliability warning if the sample is still small.
4. Only once `CalibrationResult.is_statistically_reliable` is `True`
   (and the trial data is judged representative) should
   `result.eer_threshold` be used to construct a
   `SpeakerVerifier(..., threshold=result.eer_threshold,
   is_calibrated=True)`.

## API summary

`src/speaker_verification/calibration.py`:
- `calibrate(genuine_scores, impostor_scores) -> CalibrationResult` —
  main entry point.
- `compute_far(impostor_scores, threshold) -> float`,
  `compute_frr(genuine_scores, threshold) -> float` — single-threshold
  helpers.
- `compute_far_frr_curve(genuine_scores, impostor_scores) ->
  Tuple[ThresholdOperatingPoint, ...]` — the full curve (also the ROC
  curve: FAR is the false-positive rate, `1 - FRR` is the true-positive
  rate).
- `compute_eer(genuine_scores, impostor_scores) -> (eer, eer_threshold,
  method_description)`.
- `compute_roc_auc(operating_points) -> float`.
- `compute_score_statistics(scores) -> ScoreStatistics`.

`src/speaker_verification/verifier.py`:
- `SpeakerVerifier(encoder, threshold, is_calibrated=False)` — threshold
  is a **required** argument with no default.
- `verifier.verify_embeddings(embedding_a, embedding_b) ->
  VerificationResult`
- `verifier.verify_files(reference_path, test_path) -> VerificationResult`
- `VerificationResult` always includes the raw `score`, the `threshold`
  that was applied, the `outcome` (`MATCH`/`NON_MATCH`), and whether
  `is_calibrated`.

## Explicitly out of scope for Phase 6

Fuzzy logic, ML-based classification, and ECAPA-TDNN fine-tuning are not
introduced here — see docs/architecture.md.
