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

## Phase 7: large-scale calibration using VoxCeleb

Phase 6 established the calibration *engine* (FAR/FRR/EER/ROC-AUC) and
proved it works correctly on synthetic data, but this project had only
one real genuine trial and zero impostor trials — nowhere near enough
to produce a trustworthy threshold. Phase 7 does not change the
calibration math at all; it builds a pipeline to feed
`calibration.calibrate()` real, large-scale genuine/impostor scores
from a public speaker-verification dataset, so a data-driven baseline
threshold can actually be produced.

**Phase 7 does not fine-tune or modify ECAPA-TDNN in any way.** The
pretrained `speechbrain/spkrec-ecapa-voxceleb` checkpoint and the
existing `audio.py` → `encoder.py` → `similarity.py` → `calibration.py`
pipeline are reused exactly as built in Phases 3–6. Phase 7 only adds
*more real trial data* flowing into that unchanged pipeline.

### Why VoxCeleb

[VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) is the
standard public benchmark for speaker verification: VoxCeleb1 has
1,251 speakers and 150,000+ utterances; VoxCeleb2 has 6,112 speakers
and 1,000,000+ utterances. VGG (the publisher) also distributes
established verification trial-list protocols (VoxCeleb1, VoxCeleb1-E,
VoxCeleb1-H, and "cleaned" variants), which is exactly the labeled
genuine/impostor pair structure `calibration.py` needs — rather than
this project inventing its own ad hoc pairing scheme. Using an
established, published protocol also makes any resulting numbers
comparable to other published ECAPA-TDNN/VoxCeleb results, as a sanity
check on this project's own pipeline correctness.

**Access**: VoxCeleb audio is not freely downloadable — VGG requires
completing their own registration/access process. This project does
not scrape, mirror, or bypass that process (see
`src/speaker_verification/datasets/voxceleb.py`'s module docstring).
The Phase 7 pipeline is built and tested against synthetic fixtures
that reproduce VoxCeleb's file formats; running it on real data
requires the user to obtain the trial-list file and audio separately
(see "What you need to obtain manually" below).

### Pipeline

```
VoxCeleb trial-list file (official format)
        ↓
scripts/build_trials.py
  - parse_trial_list(): text parsing only, no audio touched
  - validate_trials(): drop trials whose audio files don't exist
  - sample_trials(): deterministic, seeded, non-first-N subset
  - split_trials_by_speaker(): speaker-disjoint calibration/evaluation split
        ↓
data/voxceleb/trials/*.csv  (trial_id, reference_id, test_id,
                              reference_path, test_path, label,
                              reference_speaker, test_speaker, split)
        ↓
scripts/extract_embeddings.py
  - loads SpeakerEncoder ONCE
  - encoder.py + audio.py, exactly as in Phase 3/4 -- unmodified
  - caches embeddings by utterance id (embedding_cache.py)
        ↓
outputs/embeddings/voxceleb/  (embeddings.pt + manifest.jsonl)
        ↓
scripts/score_trials.py
  - EXISTING similarity.cosine_similarity() -- no reimplementation
        ↓
outputs/scores/voxceleb_scores.csv  (trial_id, reference, test, label,
                                      score, split, reference_speaker,
                                      test_speaker)
        ↓
scripts/run_voxceleb_calibration.py
  - EXISTING calibration.calibrate() -- called ONLY on the
    calibration-split scores
  - FAR/FRR of the resulting threshold measured on the held-out
    evaluation-split scores
        ↓
Calibration report (Phase 7J format, printed and optionally saved)
```

### Trial labels

Following VoxCeleb's own trial-list convention: `1` = genuine
(same-speaker pair), `0` = impostor (different-speaker pair). This is
the same `score >= threshold -> MATCH` / genuine-vs-impostor framing
`calibration.py` already used in Phase 6 — Phase 7 does not introduce a
new labeling convention.

### Deterministic subset sampling (Phase 7B)

VoxCeleb trial-list files are commonly ordered by speaker block (every
trial for one reference speaker appears consecutively). Taking the
first N trials would concentrate a subset on a handful of speakers and
bias anything calibrated from it. Instead, `trials.sample_trials()`
separates genuine and impostor trials, shuffles each list with a seeded
`random.Random(seed)`, and takes the first `max_genuine`/`max_impostor`
of each shuffled list. This is deterministic for a given `(trials,
seed)` pair (re-running with the same seed reproduces the exact same
subset) and is a uniform sample over whatever trials were provided —
see the docstring in `src/speaker_verification/datasets/trials.py` for
the full reasoning.

### Calibration vs. evaluation split (Phase 7F/7G)

**This is the most important methodological point in Phase 7: the
threshold is never chosen and measured on the same trials.**

`trials.split_trials_by_speaker()` assigns every unique speaker
appearing in the sampled subset to either the calibration side or the
evaluation side (deterministically, by a seeded shuffle), then keeps a
trial in the calibration set only if BOTH its speakers are on the
calibration side, in the evaluation set only if BOTH are on the
evaluation side, and **drops** any trial whose two speakers landed on
opposite sides (since including it in either set would leak a speaker
across the boundary). `scripts/run_voxceleb_calibration.py`:

1. Runs `calibration.calibrate()` — and therefore selects the EER
   threshold — using **only the calibration-split** genuine/impostor
   scores.
2. Applies that fixed threshold to the **evaluation-split** scores via
   `calibration.compute_far()` / `compute_frr()`, producing the FAR/FRR
   numbers that are reported as this baseline's actual performance.

**On the official VoxCeleb protocol itself**: the official VoxCeleb1 /
VoxCeleb1-E / VoxCeleb1-H trial lists are each a single evaluation
protocol — VGG does not ship them pre-split into "calibration" and
"evaluation" halves. The speaker-level split described above is this
project's own addition, applied on top of whichever official trial
list is supplied, specifically to avoid tuning and reporting on the
same data. If a user instead wants results directly comparable to a
published VoxCeleb1 EER number, they would run `calibration.calibrate()`
on the *entire* official trial list as a single (uncalibrated-split)
evaluation and report that separately — that is a valid, different use
of this pipeline, not what `run_voxceleb_calibration.py` does by
default.

### Speaker leakage (Phase 7G)

`trials.speaker_overlap()` independently re-checks — it does not just
trust `split_trials_by_speaker()`'s construction — whether any speaker
id appears in both the calibration and evaluation sets.
`run_voxceleb_calibration.py` always reports this count; zero overlap
is expected and required for the evaluation-set FAR/FRR to be a
meaningful, unbiased estimate. If a user supplies pre-split trial data
where the *official* protocol intentionally reuses speaker identities
across trials, that must be documented explicitly rather than silently
treated as speaker-disjoint — this project's own split
(`split_trials_by_speaker`) guarantees disjointness by construction, so
non-zero overlap should only occur if a user bypasses it.

### Embedding cache design

See the module docstring in
`src/speaker_verification/embedding_cache.py` for the full reasoning.
Summary: a consolidated `torch.save()` dict of
`{utterance_id: embedding_tensor}` plus a JSON-Lines metadata manifest,
chosen because Phase 7H's target scale (thousands, not millions, of
utterances) does not need one-file-per-utterance or a sharded/columnar
format. The cache is keyed by dataset-relative utterance id (not by
audio file path), stores embedding dimension / model source / sample
rate / extraction timestamp per entry, and never stores audio. Writes
are atomic (temp file + `os.replace`). An utterance referenced by
multiple trials is embedded exactly once (Phase 7K "duplicate
utterance handling").

### Threshold methodology (unchanged from Phase 6)

The threshold reported by `run_voxceleb_calibration.py` is the **EER
threshold from `calibration.calibrate()`**, run on real VoxCeleb
calibration-split genuine/impostor scores — the exact same, unmodified
Phase 6 algorithm described earlier in this document. It is never
0.635386 (this project's one real personal-recording score — see
above) and never any other hard-coded value. If the calibration split
ends up with zero genuine or zero impostor scores (e.g. an
under-sized `--max-genuine`/`--max-impostor` subset, or a
speaker-split that happens to starve one side), `calibrate()` raises
`InsufficientCalibrationDataError` and `run_voxceleb_calibration.py`
reports "Calibration UNAVAILABLE" — it does not fall back to any
default.

### Personal recordings stay out of VoxCeleb calibration (Phase 7L)

`scripts/score_personal_recording.py` scores this project's own two
personal recordings (kept outside Git — see `.gitignore`) using the
same pipeline, but labels the result explicitly **"Personal
same-speaker validation (OUT-OF-DOMAIN / PROJECT-SPECIFIC)."** This
score is never merged into the VoxCeleb genuine/impostor score pools
and never changes the VoxCeleb-derived threshold automatically. It
exists only as a sanity check that the pipeline runs correctly on real
project audio.

### Domain limitation: VoxCeleb vs. Hindi/Hinglish (Phase 7M)

VoxCeleb is a broad, "in-the-wild" speaker dataset, predominantly
English speech drawn from celebrity interviews. This project's actual
target domain is bilingual Hindi/Hinglish speech. **A threshold or EER
calibrated on VoxCeleb is a general-purpose baseline, not proof of
performance on Hindi/Hinglish audio.** Concretely:

1. VoxCeleb calibration (Phase 7) gives a real, data-derived *baseline*
   threshold and measured FAR/FRR/EER — a large methodological upgrade
   over the single uncalibrated 0.635386 data point, but still not
   validated on this project's actual language domain.
2. A later phase needs project-specific Indian/Hindi/Hinglish
   genuine/impostor trial data (recorded or sourced separately) run
   through this same, unmodified pipeline before any Hindi/Hinglish
   accuracy claim can be made.
3. Once both exist, a global (VoxCeleb) threshold, a domain-specific
   (Hindi/Hinglish) threshold, and — only after that comparison —
   whether a fuzzy/domain-adaptive decision layer would add value, can
   all be compared. **Fuzzy logic is explicitly NOT implemented in
   Phase 7** (see below).

### Explicitly out of scope for Phase 7

- ECAPA-TDNN fine-tuning or any modification to the pretrained model.
- Fuzzy logic or any decision layer beyond the existing
  `score >= threshold -> MATCH` rule.
- Distributed computing, a vector database, or FAISS — the embedding
  cache is a single consolidated file, and scoring is a plain Python
  loop; see Phase 7I in the Phase 7 final report for the actual
  measured throughput this achieves at the target scale.
- Automatically downloading the full VoxCeleb2 dataset (1M+
  utterances) — Phase 7H starts at ~1,000 genuine + ~1,000 impostor
  trials and only scales up after that pipeline is verified working.

### What you need to obtain manually to run this on real data

This sandbox cannot download VoxCeleb (registration-gated, and this
project does not bypass that) or the pretrained ECAPA-TDNN model
(`huggingface.co` is blocked here — see docs/model.md). To actually run
Phase 7 end-to-end on real data, from a machine with normal internet
access:

1. Register for and download VoxCeleb1 audio, plus an official
   verification trial-list file (VoxCeleb1 / VoxCeleb1-E / VoxCeleb1-H,
   cleaned or not), from
   [the official VoxCeleb site](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
2. Extract the audio so it is laid out as
   `<audio_root>/<speaker_id>/<video_id>/<utterance>.wav`.
3. Check which utterance-path convention your trial list uses (see
   [Trial-list formats](#trial-list-formats-numeric-ids-vs-speaker-names)
   below). If it names speakers rather than using `idNNNNN`, also obtain
   VGG's `vox1_meta.csv` and pass it as `--meta`.
4. Run:
   ```bash
   python scripts/build_trials.py --trials <trial_list_file> --audio-root <audio_root> \
       [--meta <vox1_meta.csv>] \
       --max-genuine 1000 --max-impostor 1000 --seed 42
   python scripts/extract_embeddings.py --trials data/voxceleb/trials/subset.csv
   python scripts/score_trials.py --trials data/voxceleb/trials/subset.csv
   python scripts/run_voxceleb_calibration.py --scores outputs/scores/voxceleb_scores.csv
   ```

## Phase 8B — First real calibration results (VoxCeleb1)

**This project now has a real, data-derived threshold.** The pipeline
described above was run end-to-end on the actual VoxCeleb1 test set on
the developer's machine (RTX 3060 Laptop GPU, CUDA 12.4).

### What was run

```
37,720 official trials  ->  28,280 kept after speaker-disjoint splitting
4,715 unique utterances ->  4,715 embeddings (0 failures)
```

Embedding extraction: 23.66 ms/file, 42.27 files/sec, 940.5 MiB peak GPU
memory, model loaded once (3.17 s).

### Results

| | Calibration split | Evaluation split (held out) |
|---|---|---|
| Genuine trials | 10,968 | 7,892 |
| Impostor trials | 6,192 | 3,228 |
| Speakers | 20 | 20 |
| Genuine score mean | 0.593241 | 0.590961 |
| Impostor score mean | 0.020799 | 0.023593 |

```
ROC-AUC:        0.998634
EER:            0.008721   (0.87%)
EER threshold:  0.252784   <-- the calibrated threshold
FAR @ threshold on held-out evaluation split: 0.011462  (1.15%)
FRR @ threshold on held-out evaluation split: 0.004815  (0.48%)
Speaker overlap between splits: 0
```

The threshold was chosen using the calibration split only and then
applied unchanged to the evaluation split, whose speakers the threshold
never saw. That FAR/FRR pair is therefore an unbiased estimate.

### How to read these numbers

**Sanity check — this matches the literature.** SpeechBrain publishes
approximately 0.80% EER for `spkrec-ecapa-voxceleb` on VoxCeleb1 test.
Reaching 0.87% through an independently written pipeline is strong
evidence that preprocessing, encoding, similarity, and calibration are
all behaving correctly end to end.

**Three limits on what this number means:**

1. **It is not the official VoxCeleb1-O protocol.** 9,440 impostor
   trials were dropped to keep calibration and evaluation
   speaker-disjoint (an impostor trial survives only if both of its
   speakers land on the same side of the split). This ran on a
   28,280-trial subset, so it is close to but not directly comparable
   with published VoxCeleb1-O figures.
2. **It is a best-case, in-domain result.** ECAPA-TDNN was trained on
   VoxCeleb — English celebrity interviews. This says nothing about
   Hindi or Hinglish performance (see docs/model.md, "Cross-lingual
   accuracy caveat"). Quote it as "VoxCeleb1 English baseline," never as
   "the system's accuracy."
3. **FAR (1.15%) exceeds FRR (0.48%) on the held-out split**, whereas at
   the EER operating point they are equal by construction. This is a
   normal generalization gap — the evaluation impostors score slightly
   higher on average (0.0236 vs 0.0208) — but it means the threshold is
   marginally permissive on unseen speakers.

### Superseded: the 0.635386 figure

Earlier phases repeatedly noted that this project had exactly one
genuine trial, scoring 0.635386, and zero impostor trials — not enough
to calibrate anything. That is now historical. **0.635386 was never used
as a threshold**, and the real calibrated value (0.252784) is
substantially lower, which is exactly why guessing a threshold from a
single same-speaker score would have been a mistake: it would have
rejected a large fraction of genuine pairs.

### Trial-list formats: numeric ids vs. speaker names

The published VoxCeleb1 verification trial lists do not all use the
same utterance-path convention, and the difference matters because only
one of them matches the on-disk audio layout.

**Form 1 — canonical / numeric id** (matches the audio tree directly):

```
1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
```

**Form 2 — speaker name** (used by the widely mirrored
`voxceleb1_test.txt` / `veri_test.txt`):

```
1 Eartha_Kitt/x6uYqmx31kE_0000001.wav Eartha_Kitt/8jEAjG6SegY_0000008.wav
```

Form 2 differs in two ways: the speaker is identified by their VGGFace1
*name* instead of their VoxCeleb1 id, and the video id and utterance
number are joined by an underscore rather than a directory separator.

The name → id mapping is external data and is **not** guessed by this
project. It comes from VGG's own `vox1_meta.csv`, a tab-separated table:

```
VoxCeleb1 ID    VGGFace1 ID    Gender    Nationality    Set
id10270         Eartha_Kitt    f         USA            test
```

`datasets/voxceleb.py` reads that file (`load_speaker_name_map`) and
normalizes form-2 ids to form 1 (`normalize_utterance_id`):

```
Eartha_Kitt/x6uYqmx31kE_0000001.wav  ->  id10270/x6uYqmx31kE/00001.wav
```

Two details that are easy to get wrong, and are covered by tests:

- The video id is split on the **last** underscore, because VoxCeleb
  video ids can themselves contain underscores (e.g. `5sJomL_D0_g`).
- The utterance number is **re-padded**, not truncated: the trial list
  pads to 7 digits (`0000001`) while the audio tree uses 5 (`00001.wav`).

Parsing normalizes every id to form 1, so downstream stages (embedding
cache keys, scores CSV, calibration report) always see one consistent
identifier regardless of which list was used. Passing `--meta` with a
form-1 list is harmless. Omitting it with a form-2 list raises a clear
error naming the offending line rather than silently mis-resolving paths.
