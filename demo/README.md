# Demo recordings

Clips for showing both stages of B.S. Detector live, in Hindi and English. Drop
them into the app (`python run.py`, then *Check a voice in two steps*). Every
score below came from the app's own code (`python scripts/build_demo.py --score`),
and `manifest.json` has the full numbers.

These clips were chosen to be classified correctly. They show how the system
works; they are not a measure of how often it is right. For that, quote the
held-out results in `docs/speaker-verification-hindi-english.md` and
`docs/deepfake-improvement-plan.md`.

## 1. Real vs generated (`deepfake/`)

Put a clip in **Voice to check** alone and press *Check this voice*. Leave the
window at the start of the clip (see the note at the end). Scores are from the
silence-invariant Hindi head the app loads by default, on the first 4.04 s.

| File | What it is | Chance generated | Verdict |
|---|---|---|---|
| `hindi/real_person1.wav` | Real Hindi speaker (Common Voice) | 0.0% | real |
| `hindi/fake_xtts_clone_of_person1.wav` | XTTS-v2 clone of **that same person**, new sentence | 81.3% | generated |
| `hindi/real_person2.wav` | Real Hindi speaker (Common Voice) | 0.0% | real |
| `hindi/fake_xtts_clone_of_person2.wav` | XTTS-v2 clone of that same person | 96.3% | generated |
| `hindi/fake_edge_tts_madhur_1.wav` | Microsoft Edge neural TTS, hi-IN Madhur | 98.8% | generated |
| `hindi/fake_edge_tts_swara_2.wav` | Microsoft Edge neural TTS, hi-IN Swara | 98.2% | generated |
| `english/real_person1.wav` | Real Indian-English speaker (Common Voice) | 1.4% | real |
| `english/fake_xtts_clone_of_person1.wav` | XTTS-v2 clone of that same person | 71.7% | generated |
| `english/real_person2.wav` | Real Indian-English speaker (Common Voice) | 0.1% | real |
| `english/fake_xtts_clone_of_person2.wav` | XTTS-v2 clone of that same person | 91.2% | generated |
| `english/fake_edge_tts_prabhat_1.wav` | Microsoft Edge neural TTS, en-IN Prabhat | 100.0% | generated |
| `english/fake_edge_tts_prabhat_2.wav` | Microsoft Edge neural TTS, en-IN Prabhat | 97.7% | generated |

**The interlock, live:** put `real_personN.wav` in **Known voice** and its
`fake_xtts_clone_of_personN.wav` in **Voice to check**. The deepfake check
blocks the comparison. Swap them (clone as the known voice) and it is blocked
again, because the known voice is checked too. Without the interlock, the speaker
check alone would accept most clones as the real person: it did for 19 of 20 in
our tests.

## 2. Same person or not (`speaker/`)

Put the first file in **Known voice**, the second in **Voice to check**, and
press *Run both checks*. "Old" is the VoxCeleb cut-off the app used before
Phase 9, recorded with the original ECAPA. Since the app now loads the
fine-tuned ECAPA, `build_demo.py --score` no longer reproduces that column.

| Language | Known voice | Voice to check | Truth | New result (score vs cut-off) | Old result |
|---|---|---|---|---|---|
| Hindi | `person_A_take1` | `person_A_take2` | same | **Same speaker** (4.77 vs 2.82) | same |
| Hindi | `person_A_take1` | `person_B` | different | **Different speaker** (0.33 vs 2.50) | same ✗ |
| Hindi | `same_room_person_A` | `same_room_person_B` | different | **Different speaker** (0.57 vs 2.16)* | same ✗ |
| English | `person_A_take1` | `person_A_take2` | same | **Same speaker** (4.21 vs 2.77) | same |
| English | `person_A_take1` | `person_B` | different | **Different speaker** (0.59 vs 2.77) | same ✗ |
| English | `same_room_person_A` | `same_room_person_B` | different | **Different speaker** (1.15 vs 2.70) | same ✗ |

The `same_room_*` pairs are two different people put through the same simulated
room and laptop microphone: the case where the old version called two friends
the same person. *`hindi/same_room_person_B.wav` has 7.3 s of speech, so the app
adds that results under 8 s are less certain.

Each speaker file is 3-5 short questions by one person, joined (10-20 s).

## Things to know before presenting

- **Keep the deepfake window at 0 s.** The app puts it there. Moved into the
  middle of a sentence, it can flag a real voice: started 1 s into the speech,
  the detector flagged 7 of our 18 real files. See
  `docs/deepfake-improvement-plan.md`.
- Edge-TTS voices were the old head's weak spot (10 of 16 caught at the 0.5 line).
  All four here are caught.
- For your own live recordings, speak for 10-15 s per clip.

## Sources and licences

| Files | Source | Licence |
|---|---|---|
| `deepfake/*/real_person*.wav` | Mozilla Common Voice 17 (via `fixie-ai/common_voice_17_0`); clip ids in `manifest.json` | CC0-1.0 |
| `speaker/**` | Google Simple Voice Questions (`google/svq`), `hi_in` / `en_in`, clean condition | CC-BY-4.0 |
| `deepfake/*/fake_xtts_*.wav` | Generated with Coqui XTTS-v2, cloning the matching Common Voice clip | Coqui Public Model License: non-commercial use |
| `deepfake/*/fake_edge_tts_*.wav` | Generated with Microsoft Edge neural TTS (`edge-tts`) | Microsoft service terms |

The cloned voices are of anonymous Common Voice contributors who released their
recordings under CC0, used here only to demonstrate deepfake detection. All
generated clips are labelled as generated, here and in `manifest.json`.

Rebuild: `python scripts/make_fakes.py` (in an environment with coqui-tts and
edge-tts) then `python scripts/build_demo.py --real <dir> --fake <dir> --score`.
