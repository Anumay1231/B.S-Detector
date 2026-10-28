# Stage 1 (deepfake detection): measured gaps and improvement plan

Status: the app ships XLS-R 300M with the silence-invariant Hindi head
(`models/sls_head_silence_invariant.pth`, described below). The numbered plan
items have not been applied yet.

## What we measured (October 2026)

`scripts/deepfake_check.py` on clips the head never saw in training: real
Common Voice 17 speakers (CC0) and fakes made by `scripts/make_fakes.py`.
Real clips were also replayed through three simulated rooms with a laptop-mic
colouration and noise bed (the same rooms as the speaker benchmark).

| Clips | n | English checkpoint | Hindi head (shipped) |
|---|---|---|---|
| Real Hindi | 62 (+186 in rooms) | 1.6% flagged | **0% flagged** |
| Real Indian English | 24 (+72 in rooms) | 4.2% flagged | **0% flagged** |
| XTTS-v2 clone of a real speaker, Hindi | 10 | 30% caught | **100% caught** |
| XTTS-v2 clone of a real speaker, English | 10 | 10% caught | **100% caught** |
| Edge-TTS stock voice, Hindi (Madhur, Swara) | 8 | 12.5% caught | **75% caught** |
| Edge-TTS stock voice, English (Prabhat, Neerja) | 8 | 0% caught | **50% caught** |

Read: no false alarms on real speech in either language, even through a room;
every voice clone caught. The gap is commercial neural TTS: Edge's Neerja voice
was missed every time (p_spoof 0.26-0.35). Separately, one LibriSpeech
audiobook reader (`frontend/public/samples/speaker-b-1.wav`) is flagged at 0.91,
so heavily processed studio English can still trip it.

### The head leans on leading silence (the most important finding)

The same 36 fakes and 20 real clips, edited two ways:

| | Leading/trailing silence trimmed | 0.5 s of quiet room tone added in front |
|---|---|---|
| Fakes flagged | 100% | **6%** |
| Real clips flagged | **35%** | 0% |

And on real clips, moving the 4.04 s window from 0 s to 1 s into the clip (so it
starts mid-speech) flips most of them to "generated" (p_spoof 0.7-1.0); putting
0.5 s of silence back in front restores ~0.00 every time.

So the Hindi head has largely learnt *does the clip open on a pause?*: bona fide
corpus recordings start with silence, TTS output starts on the first phoneme.
The table above is therefore only valid for clips as TTS engines save them. A
deepfake with half a second of silence in front passes; real speech cut
mid-word is flagged. Two consequences in the app today:
- the window must stay at the start of the clip (dragging it later can flag a
  real voice), and
- the interlock checks only the first window of each clip, not every window.

**IndicSynth head (the default until 28 Oct).** The team's head fine-tuned on
IndicSynth clones (`models/sls_head_n500_indicsynth_best.pth`) was run through
the same edits (`outputs/indicsynth_check.json`), % flagged at p_spoof >= 0.5:

| | Old N=500 head | IndicSynth head |
|---|---|---|
| Fakes as saved (36) | 83.3% | **91.7%** |
| Real as saved (86) | 0% | 0% |
| Fakes, 0.5 s quiet in front | 5.6% | 16.7% |
| Real, window starting 1 s in | 68.6% | 76.7% |

It catches more fakes as saved, with no new false alarms, so the app loaded it
next. The silence shortcut is unchanged. Its training zero-pads short clips while
the app tile-pads them, and neither randomises leading silence.

The fix is item 1 below, with leading-silence randomisation as part of it:
trim every training clip and re-pad it with a random 0-1 s of silence or room
tone, for both classes, so silence carries no label information.

### Silence-invariant head (the app's default since 28 Oct)

`models/sls_head_silence_invariant.pth` (`hindi-robust`, `scripts/train_sls_robust.py`)
was trained with random 0-0.8 s of silence or room tone in front and windows
cut mid-utterance. When this was measured, the browser snapped the window to
the speech onset; long clips are scored over 4.04 s windows with a 2 s stride.
Measured with
`scripts/ui_path_check.py` on the 22 demo clips plus the 5 bundled samples
(18 real, 9 fake):

| Condition | Real flagged | Fakes caught |
|---|---|---|
| Browser window (snapped to speech onset) | 6/18 | 9/9 |
| First 4.04 s from 0 s | 0/18 | 9/9 |
| Silence trimmed | 7/18 | 9/9 |
| 0.5 s quiet in front | 7/18 | 9/9 |
| Starts 1 s into speech | 7/18 | 8/9 |
| Whole file, worst window | 10/18 | 9/9 |
| Whole file, mean of the 2 worst windows (now the default) | 8/18 | 9/9 |

- **Silence shortcut:** largely gone. Adding or removing silence now barely
  changes the answer, and fakes are caught in every condition.
- **New problem:** every false alarm is a real Google SVQ phone recording, and
  on those most windows score high, not just one. Clean Common Voice and FLEURS
  clips pass. The head's only real training speech was FLEURS Hindi. Because
  the voice to check is flagged, 4 of the 6 demo speaker pairs are blocked by
  the interlock in the browser.
- **Aggregation:** the mean of the 2 worst windows replaced the worst window
  (10 to 8 of 18 real files flagged, all 9 fakes and 4/4 real-then-clone
  splices still caught). Requiring 2 flagged windows missed a splice; the
  median or mean missed two. No rule fixes the phone recordings.
- **Reported numbers:** the 98.86% accuracy and 100% per Edge voice in
  `outputs/sls_robust_results.json` come from a split that shares speakers and
  Edge voices between train and test, with the checkpoint chosen on the test
  set, so they are not held-out results. The 99.0% on SEA-Spoof is not in any
  results file in the repo.
- **Phone line** (`scripts/ui_path_check.py --phone`: 8 kHz, 300-3400 Hz,
  8-bit mu-law): fakes caught in the browser window drop from 9/9 to 6/9 (three
  English fakes at p_spoof 0.47-0.50), real false alarms drop from 6/18 to 3/18,
  and the speaker check returns "not sure" on both same-room pairs and on the
  English clone pairs that now get past stage 1 (it does not accept the clone).
  12 wrong in total, against 10 on normal audio.
- **Next:** retrain with real phone-style and English speech (SVQ, Common
  Voice) plus codec augmentation, split by speaker and voice, with a separate
  validation set (item 1 below). The target is `scripts/ui_path_check.py`
  reporting 0 wrong.

**Update, 29 Oct: window back at 0 s.** Snapping to the speech onset was what
caught the SVQ recordings: from 0 s the same head flags none of them. The
browser now starts the window at 0 s for loaded and recorded clips, and
`ui_path_check.py` sends what the browser sends. Results with the same head:

| Check | Onset window (before) | 0 s window (now) |
|---|---|---|
| `ui_path_check.py`, normal audio | 10 wrong | 0 wrong |
| `ui_path_check.py --phone` | 7 wrong | 7 wrong |

The phone run on 29 Oct gave 7 wrong before the change, not the 12 above. All
7 remain: three English fakes at p_spoof 0.47-0.50 get through, both
same-room pairs come back "not sure", and so do the two English clone pairs
(the speaker check still does not accept the clone). No real phone-line
file is flagged. The stress table is unchanged, so the whole-file top-2
score still flags 8/18 real files. Anything that scores every window, such
as an interlock over the whole clip, has to wait for the retrained head.

Also measured: ECAPA scores **19 of 20 XTTS clones as the real person** at the
old VoxCeleb cut-off (cosine 0.30-0.67), so the speaker check alone cannot stop
a clone. This is why the interlock now also checks the known voice, not just the
voice to check.

## Plan, in order of payoff

0. **Remove the silence shortcut.** Re-pad every training clip, real and fake,
   with a random 0-1 s of silence / room tone after trimming, and add windows that
   start mid-utterance. Re-test with the two edits above; the gap between the
   columns should close. Until then, report stage 1 results with this caveat.
1. **Fine-tune the SLS head on more generators.** Add Edge-TTS (all hi-IN and
   en-IN voices, plus en-US / en-GB), XTTS-v2 clones and English bona fide
   speech (Common Voice, SVQ) to the N=500 Hindi set. Head only, backbone frozen,
   as in the benchmark. Protocol: hold out whole voices (e.g. Neerja, Swara) and
   whole speakers for testing, so the gain is measured on unseen voices.
   *Blocker:* the SEA-Spoof test split behind the reported 98% is gated on
   Hugging Face. It needs a team member's HF token to re-run, so we can confirm
   the Hindi benchmark does not drop after fine-tuning.
2. **Codec and channel augmentation** in that fine-tune (MP3 / Opus re-encoding,
   room impulse responses, browser noise suppression), since live clips arrive
   through a mic and a browser.
3. **Score the whole clip, not one window**, once item 0 makes later windows
   trustworthy. Today only the first window of each clip is safe to score, so a
   clip that is real for 4 s and cloned after that is not caught.
4. **Calibrate the 0.5 decision threshold** on a Hindi + English dev set rather
   than using the softmax midpoint.
5. **Ensemble with a second detector** (e.g. an AASIST-style model) if the
   fine-tuned head still misses a TTS family.
