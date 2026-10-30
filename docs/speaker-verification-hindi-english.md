# Phase 9: Speaker verification on Hindi and Indian-English speakers

Phase 8 calibrated the ECAPA-TDNN speaker check on VoxCeleb1 (EER 0.87%,
threshold 0.2528). Phase 7M flagged the open question: does that hold for Hindi
and Hinglish speakers? In live use it did not. Two different people recorded on
the same laptop came out as "same speaker". This phase measures why and fixes it.

Scripts: `scripts/sv_benchmark.py`, `scripts/sv_duration.py`,
`scripts/sv_segments.py`, `scripts/sv_fusion.py`. Results:
`outputs/sv_bench/fusion_fit.json`. Code: `server/speaker.py`,
`src/speaker_verification/wespeaker.py`.

## 1. Data

All public and speaker-labelled. Train and test speakers never overlap.

| Corpus | Use | Speakers | Licence |
|---|---|---|---|
| Mozilla Common Voice 17, Hindi (`validated`, split 50/50 by speaker hash) | fitting | 139 | CC0 |
| Mozilla Common Voice 17, English (`train`) | fitting | 600 | CC0 |
| Google SVQ `hi_in`, `en_in` (phone recordings: clean, background speech, media noise, traffic) | test, and fitting in cross-validation | 18 hi, 23 en (the same people record both languages) | CC-BY-4.0 |

**Same-room condition.** The failure case is two people on one laptop mic in one
room. Each trial of that condition passes *both* clips through the same simulated
room: a synthetic reverberant impulse response (RT60 0.35-0.8 s), a microphone
band-pass colouration and one noise bed at 12-20 dB SNR. Genuine and impostor
trials get identical treatment, so a model that embeds the room instead of the
voice shows up as rising impostor scores.

Impostors are always the same gender as the target (the harder, realistic case).
Clips are 1-3 utterances of one person joined, giving 3-20 s of speech.

## 2. Root cause

On held-out SVQ speakers, the VoxCeleb threshold accepts **42-70% of
same-gender impostors** (table in section 5). Different Indian speakers score
~0.2 cosine on average (VoxCeleb impostors: 0.02), and the score rises with clip
length. So the longer two friends talked, the more surely the old system called
them the same person. The threshold, not the recording, was wrong for this
population.

## 3. What we tried, and what each step bought

Per-utterance SVQ trials, equal error rate (lower is better):

| ~seconds per side | ECAPA | WeSpeaker ResNet34 | ECAPA + ResNet34 fused |
|---|---|---|---|
| Hindi, 5 s, clean / room | 7.5% / 15.9% | 5.4% / 15.4% | **4.8% / 13.7%** |
| Hindi, 15 s, clean / room | 0.9% / 2.8% | 0.05% / 2.8% | **0.0% / 1.9%** |
| English, 5 s, clean / room | 13.9% / 20.2% | 12.1% / 22.9% | **11.8% / 19.6%** |
| English, 15 s, clean / room | 3.0% / 4.3% | 3.7% / 6.7% | **2.4% / 4.3%** |

1. **Speech length dominates.** Going from 5 s to 15 s per side cut ECAPA's
   same-room error from 15.9% to 2.8% (Hindi). The app now needs 3 s of speech
   minimum, records up to 30 s, and marks results under 8 s as less reliable.
2. **Segment averaging.** Each clip is cut into 3 s pieces, each piece embedded,
   and the normalised embeddings averaged. On the same held-out clips this beat
   one pass over the whole clip (Hindi 11 s+ same-room EER 16.8% to 5.8%).
3. **Score fusion with a second model.** WeSpeaker ResNet34-LM (VoxCeleb2, ONNX)
   alone is about as good as ECAPA, but makes different mistakes. Each cosine is
   z-normalised by its impostor mean/std and the two are averaged, giving 10-30%
   fewer errors than ECAPA alone. ECAPA stays the primary model; if
   onnxruntime is missing the app falls back to ECAPA with its own constants.
4. **Duration-aware threshold.** The EER threshold rises with speech length
   (ECAPA cosine 0.34 at 3 s to 0.52 at 12 s), so it is fitted as
   `t(d) = a + b ln d`, with d the speech in the shorter clip.
5. **Multi-corpus calibration.** Common Voice alone set the threshold too low for
   SVQ's phone audio (0.47 vs ~0.55 at 9 s). Fitting on Common Voice plus half of
   the SVQ people and testing on the other half (2-fold, swapped) fixed it.
6. **An UNCERTAIN verdict.** Scores within `max(0.40, 1.40 - 0.60 ln d)` (fused
   z units) of the threshold return "not sure, record more" instead of a guess.
   Without the 0.40 floor the band vanishes by 8 s, where same-room Hindi was
   still only ~91% right.

Also tried and not shipped: LDA and AS-norm backends trained on Common Voice
(`src/speaker_verification/backend.py`). Mean-centring + AS-norm helped clean
5 s clips (Hindi EER 7.5% to 5.7%) but hurt long same-room trials, and LDA
over-fitted 139 Hindi speakers. ECAPA2, WavLM-SV and ResNet293 runs were started
but not finished in time.

## 4. Shipped decision

```
score   = mean( (cos_ecapa - 0.1860)/0.1298 , (cos_resnet34 - 0.2207)/0.1384 )
t(d)    = 0.0189 + 1.0776 ln d          d = speech seconds in the shorter clip, 3..20
m(d)    = max(0.40, 1.4006 - 0.6043 ln d)
MATCH if score >= t + m;  NON_MATCH if score < t - m;  otherwise UNCERTAIN
```

## 5. Results (held out, 2-fold cross-validation over SVQ people)

Balanced accuracy = mean of (genuine accepted, impostor rejected).

| SVQ set | Condition | Speech | Old: impostors accepted at 0.2528 | Old accuracy | ECAPA + duration threshold | Fused | **Fused + unsure: accuracy on answers** | Answered | Pairs (gen / imp) |
|---|---|---|---|---|---|---|---|---|---|
| Hindi | clean | 3-8 s | 42.5% | 78.8% | 95.0% | 96.2% | **98.4%** | 91% | 195 / 751 |
| Hindi | clean | 8 s+ | 68.6% | 65.7% | 98.8% | 99.9% | **100.0%** | 98% | 74 / 337 |
| Hindi | same room | 3-8 s | 45.0% | 76.5% | 87.7% | 87.9% | **92.1%** | 83% | 394 / 1504 |
| Hindi | same room | 8 s+ | 70.4% | 64.8% | 91.8% | 90.6% | **96.9%** | 81% | 90 / 438 |
| English | clean | 3-8 s | 42.7% | 77.6% | 93.9% | 94.0% | **96.6%** | 86% | 273 / 1583 |
| English | clean | 8 s+ | 51.6% | 73.4% | 96.4% | 97.0% | **98.1%** | 95% | 92 / 410 |
| English | same room | 3-8 s | 45.5% | 76.7% | 89.8% | 89.1% | **92.6%** | 81% | 549 / 3163 |
| English | same room | 8 s+ | 51.5% | 74.2% | 97.1% | 97.8% | **99.0%** | 93% | 95 / 345 |

With 8 s or more of speech in each clip, the new check is right on 96.9-100% of
the pairs it answers, in both languages, clean or sharing a room. It asks for a
longer recording on 2-19% of pairs. Below 8 s it is 92-98% right and the UI
says the result is less reliable.

**Limits.** SVQ has 41 speaker-language sets, so pair counts are modest (see last
column); the same-room condition is simulated, not recorded; Hinglish
(code-mixed) speech is not separately tested.

## 6. Voice clones and the interlock

XTTS-v2 clones of 20 real Common Voice speakers, scored against a different
clip of that speaker: ECAPA called **19 of 20 the real person** at the old
threshold (cosine 0.30-0.67), and a stricter threshold still passes about half.
Speaker verification cannot stop a clone by itself. The interlock now runs the
deepfake check on the known voice as well as the voice to check, so a cloned
reference cannot vouch for a clone. See `docs/deepfake-improvement-plan.md` for
stage 1's own limits.

## 7. The Indian-adapted ECAPA, tested fairly (30 Oct)

`models/ecapa_indian_adapted.pth` was fine-tuned on SVQ fold 0 (`fold_of() == 0`
in `scripts/sv_fusion.py`) and Common Voice Hindi part 0. Until 30 Oct the app
loaded it automatically, with `FUSION_NORM` / `DECISION` still fitted on the
original ECAPA: a pairing nobody had measured.

**Clean and room audio.** `python scripts/sv_fusion.py --fair --ecapa <model>` fits
the fused constants on Common Voice (train halves) + SVQ fold 0 for each ECAPA,
then reports only on SVQ fold 1 and Common Voice Hindi part 1. Embeddings come
from one pass that feeds identical audio to all three models
(`sv_duration.py embed --model ecapa --also ecapa_adapted wespeaker_r34`).
Accuracy on the pairs the fused check answers, adapted minus original:

| Set | Cells better or equal | Cells worse | Range |
|---|---|---|---|
| SVQ Hindi | 7 of 7 | 0 | +0.0 to +1.9 points |
| Common Voice Hindi part 1 | 7 of 8 | 1 | -0.03 to +0.4 |
| SVQ Indian English | 4 of 9 | 5 | -1.05 to +0.2 |

The ship rule was "keep it only if no cell gets worse", so the app now loads
the **original ECAPA** again, matching its constants (`server/speaker.py`;
`BSD_ECAPA_WEIGHTS` points at other weights). The adapted file stays in
`models/` for a comparison on real recordings (`scripts/score_recordings.py`).

**Telephone line.** `scripts/sv_phone_fair.py` repeats the fair test after the
phone line of `scripts/ui_path_check.py --phone` (8 kHz band, 8-bit mu-law),
alone and after a simulated room. **The check is not reliable on phone audio
with either model.** Share of different-person pairs accepted as the same
person, 8 s+ of speech:

| Set | Condition | Original | Adapted |
|---|---|---|---|
| SVQ Hindi | phone | 16.8% | 8.6% |
| SVQ Hindi | room + phone | 52.0% | 63.6% |
| SVQ English | phone | 21.0% | 19.8% |
| SVQ English | room + phone | 27.9% | 36.1% |
| Common Voice Hindi | phone | 3.5% | 2.6% |
| Common Voice Hindi | room + phone | 13.5% | 15.7% |

With 3-8 s of speech it is 30-58% on SVQ. The constants were fitted on
full-band audio, and the phone band leaves less to tell voices apart, so
different people score higher. Neither model fixes this.

**What the app does now.** `server/speaker.py` measures how much of each clip's
speech energy lies above 4 kHz. Below 2e-5, the clip counts as phone-band, and the
pair is answered UNCERTAIN (the score is still reported, and the response lists the
clips in `phoneBand`). A simulated phone line measures at most 3e-6. Demo and sample
files measure at least 3e-3, and real dataset clips have a median of 1.5e-3 to
9e-3; the few real clips below 2e-5 are low-passed like a phone line, so the same
caution applies to them. `ui_path_check.py --phone` went from 6 wrong (two
same-room pairs accepted as one person) to 2, both stage 1 misses. Fitting separate
phone-band constants is possible, but even then room + phone pairs reached only
55-64% accuracy above, so "not sure" is the honest answer for now.

**Short same-room clips, re-measured (30 Oct).** From the same fair test, with
the shipped setup (original ECAPA, fused), accuracy on the pairs it answers,
with the share answered in brackets:

| Same-room pairs | SVQ Hindi (fold 1) | SVQ English (fold 1) | Common Voice Hindi part 1 |
|---|---|---|---|
| 3-6 s of speech | 83.6% (76%) | 82.8% (72%) | 97.6% (93%) |
| 6-8 s | 92.7% (78%) | 90.2% (78%) | 99.6% (96%) |

The 92% in section 5 was one 3-8 s bucket, cross-validated over both SVQ folds.
Split finer and tested on unseen people only, the shortest SVQ clips sit near
83%. The UI already says results under 8 s are less certain; for same-room
recordings of under 6 s, that warning matters most.
