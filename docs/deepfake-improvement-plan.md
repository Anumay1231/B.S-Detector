# Stage 1 (deepfake detection): measured gaps and improvement plan

Status: **plan only.** Nothing below has been applied to the shipped detector
(XLS-R 300M + Hindi-adapted SLS head, `models/sls_head_n500_seed2_best.pth`).

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

**IndicSynth head (now the app's default).** The team's head fine-tuned on
IndicSynth clones (`models/sls_head_n500_indicsynth_best.pth`) was run through
the same edits (`outputs/indicsynth_check.json`), % flagged at p_spoof >= 0.5:

| | Old N=500 head | IndicSynth head |
|---|---|---|
| Fakes as saved (36) | 83.3% | **91.7%** |
| Real as saved (86) | 0% | 0% |
| Fakes, 0.5 s quiet in front | 5.6% | 16.7% |
| Real, window starting 1 s in | 68.6% | 76.7% |

It catches more fakes as saved, with no new false alarms, so the app now loads
it. The silence shortcut is unchanged. Its training zero-pads short clips while
the app tile-pads them, and neither randomises leading silence.

The fix is item 1 below, with leading-silence randomisation as part of it:
trim every training clip and re-pad it with a random 0-1 s of silence or room
tone, for both classes, so silence carries no label information.

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
