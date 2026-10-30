# Stage 1 (deepfake detection): measured gaps and improvement plan

Status: the app ships XLS-R 300M with the phase 12 Hindi head
(`models/sls_head_phase12.pth`, see "Phase 12 head" below). Plan items 0-2 are
applied in it; items 3-5 are not.

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

### Phase 12 head (the app's default since 30 Oct)

**Training** (`scripts/train_phase12.py`, data by `phase12_data.py` and
`phase12_fakes.py`). Starting from the silence-invariant head, backbone frozen:

- Real: SVQ hi/en in all four environments, Common Voice hi/en (Indian-accented
  English as its own source), FLEURS hi, IndicTTS hi (studio), Svarah (Indian
  English), Lahaja and Kathbath (Hindi).
- Fake: IndicSynth (XTTS-v2, FreeVC24), Edge-TTS stock voices, and XTTS-v2
  clones of Common Voice speakers.
- Augmentation, the same for both classes: the window starts at 0 s, mid-sentence
  or after 0-1 s of added quiet; then a phone line, MP3 or Opus, background noise,
  a random room or a loudspeaker replay, and a random gain.
- Splits are by person and by TTS voice (`phase12_common.py`). Neerja and Swara
  are test-only, two more Edge voices are val-only, and demo speakers are in no
  split at all. The epoch was chosen on val.

**Held-out results** (`scripts/eval_phase12.py`): 1,782 windows of people and
voices no head was trained or selected on, each heard under one condition.
Balanced accuracy at the app's 0.5 cut-off:

| Condition | Silence-invariant | IndicSynth | Channel-robust | Retrained alone | **Shipped (0.7/0.3 blend)** |
|---|---|---|---|---|---|
| **All** | **73.4%** | **67.1%** | **81.1%** | **84.0%** | **76.3%** |
| From 0 s | 90.0% | 77.8% | 94.7% | 94.6% | 92.8% |
| Phone line | 80.2% | 70.8% | 88.9% | 87.4% | 83.3% |
| MP3 | 68.4% | 61.7% | 85.0% | 88.3% | 76.2% |
| Opus | 94.4% | 78.2% | 94.7% | 93.3% | 96.8% |
| Mid-sentence | 85.7% | 64.5% | 84.6% | 93.7% | 89.1% |
| Added quiet | 85.0% | 67.4% | 84.4% | 87.1% | 87.4% |
| Noise | 52.9% | 78.3% | 73.6% | 77.6% | 56.5% |
| Room | 49.6% | 51.3% | 58.5% | 67.1% | 50.0% |
| Loudspeaker replay | 50.0% | 50.8% | 62.4% | 63.8% | 50.0% |

Share of each fake source caught, same windows:

| Fake source | Silence-invariant | IndicSynth | Channel-robust | Retrained alone | Shipped blend |
|---|---|---|---|---|---|
| Edge-TTS, unseen voices (Neerja, Swara) | 44.4% | 41.2% | 72.5% | 59.4% | 46.2% |
| XTTS-v2 clones, English | 40.0% | 80.0% | 78.0% | 90.0% | 50.0% |
| XTTS-v2 clones, Hindi | 45.0% | 77.5% | 75.0% | 90.0% | 50.0% |
| IndicSynth XTTS-v2 | 30.4% | 69.6% | 65.2% | 82.6% | 47.8% |
| IndicSynth FreeVC24 | 65.5% | 68.5% | 78.0% | 77.5% | 67.5% |

On real speech, the shipped blend calls 92.5-100% of each source real, apart
from SVQ clean at 75% (the silence-invariant head also gets 75%). The
channel-robust head drops to 50% on SVQ clean and 80.5% on Common Voice
English. The IndicSynth head flags most of Lahaja (29% called real) and
Svarah (44%).

**What this says:**

- **The heads miss most fakes they have not seen.** The silence-invariant head
  caught 9/9 demo fakes but only 30-45% of new XTTS clones. The demo fakes are
  close to its training data, so the demo set alone overstates it. The 98%
  SEA-Spoof figure is still unchecked (the set is about 80 GB across languages).
- **The retrained head generalises best (84.0%) but did not ship.**
  `ui_path_check.py` gave 6 wrong: it flags the two LibriSpeech samples (clean US
  audiobook speech, a domain in none of the training sources) and misses Edge
  Prabhat 2 and Swara.
- **The shipped head is a weight blend** (`scripts/mix_heads.py`, 0.7 x
  silence-invariant + 0.3 x retrained). It keeps the old head's answers on
  unfamiliar real speech: every real source is equal or better except
  non-Indian Common Voice English (92.5% vs 93.5%, 2 of 200 windows), and
  `ui_path_check.py` gives 0 wrong. The phone line drops from 7 wrong to 4, and
  held-out accuracy goes from 73.4% to 76.3%. It gives back most of the room,
  replay and noise gains, though.
- **The channel-robust head** scores 81.1% held out, second best. But it got 17
  wrong on the app check, flagging the clean built-in samples. It stays last.
- **Training was unstable** at lr 3e-5: val accuracy swung from 87.7% (epoch 7)
  to 65.7% (epoch 8). Keeping the best val epoch protects the result; a lower
  learning rate would be worth trying.
- **Stress table:** from 0 s it is 0/18 real flagged and 9/9 fakes caught.
  Starting 1 s into speech, 7/9 fakes are caught (the old head: 8/9). The whole
  file still flags 9/18 real files.

**Next:** add clean read English (LibriSpeech, keeping reader 777 out, since
the bundled samples come from that reader) and retrain at a lower learning
rate. The aim is a head that ships on its own and keeps the 84% held-out
accuracy.

## Plan, in order of payoff

0. **(Done: silence-invariant head, then phase 12.) Remove the silence shortcut.** Re-pad every training clip, real and fake,
   with a random 0-1 s of silence / room tone after trimming, and add windows that
   start mid-utterance. Re-test with the two edits above; the gap between the
   columns should close. Until then, report stage 1 results with this caveat.
1. **(Done in phase 12, held-out numbers above.) Fine-tune the SLS head on more generators.** Add Edge-TTS (all hi-IN and
   en-IN voices, plus en-US / en-GB), XTTS-v2 clones and English bona fide
   speech (Common Voice, SVQ) to the N=500 Hindi set. Head only, backbone frozen,
   as in the benchmark. Protocol: hold out whole voices (e.g. Neerja, Swara) and
   whole speakers for testing, so the gain is measured on unseen voices.
   *Blocker:* the SEA-Spoof test split behind the reported 98% is gated on
   Hugging Face. It needs a team member's HF token to re-run, so we can confirm
   the Hindi benchmark does not drop after fine-tuning.
2. **(Done in phase 12.) Codec and channel augmentation** in that fine-tune (MP3 / Opus re-encoding,
   room impulse responses, browser noise suppression), since live clips arrive
   through a mic and a browser.
3. **Score the whole clip, not one window**, once item 0 makes later windows
   trustworthy. Today only the first window of each clip is safe to score, so a
   clip that is real for 4 s and cloned after that is not caught.
4. **Calibrate the 0.5 decision threshold** on a Hindi + English dev set rather
   than using the softmax midpoint.
5. **Ensemble with a second detector** (e.g. an AASIST-style model) if the
   fine-tuned head still misses a TTS family.
