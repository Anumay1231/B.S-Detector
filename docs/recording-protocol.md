# Recording protocol: real-world test set

Every "same room" number so far comes from a simulated room. This protocol
collects a small set of real recordings to test both stages the way the app is
actually used: two people, one laptop or phone, a real room.

## Who and how many

- 20-30 people, with a mix of men and women. Each person must be recorded with
  their written consent (template below).
- Everyone records in Hindi and in English. Hinglish is welcome as a third
  language if they speak it naturally.

## What each person records

For each language: two takes of about 15 s each, in each of the four setups
below. That makes 16 clips per person for Hindi and English.

| Setup | Device | Room |
|---|---|---|
| 1 | Laptop built-in mic, through the app's Record button (Chrome) | Quiet room |
| 2 | Laptop built-in mic, through the app's Record button (Chrome) | Noisy room (TV, fan or people talking) |
| 3 | Phone voice-recorder app, held normally | Quiet room |
| 4 | Phone voice-recorder app, held normally | Noisy room |

- Speak naturally. Read something, or answer a question such as "describe your
  day" or "how do you get to college?". Don't read the same sentence in every
  take.
- Record two people in the **same session, room and device** wherever possible:
  that pair is the case that matters most.
- Don't trim or edit the files.

## File names

`<person>_<lang>_<setup>_<take>.wav` (or `.m4a` / `.webm` as the device saves it).
For example: `p07_hi_2_b.wav` is person 7, Hindi, setup 2, take b.

Keep a sheet `people.csv` with `person, gender, age band, native language,
consent signed (yes/no)`. Never put names in the file names.

## Storage and privacy

- Keep the recordings and `people.csv` **off GitHub** and off any public drive.
  Use `data/real_recordings/`, which is gitignored.
- Delete everything at the end of the project, or when a participant asks.
- Use the recordings only to test B.S. Detector. Don't train on one person's
  clips and test on the same person.

## What to measure

Score the folder with the app's own code: `python scripts/score_recordings.py --people data/real_recordings/people.csv`.

- **Speaker check:** every same-person pair (take a vs b, across setups) and
  every different-person pair of the same gender, grouped by language, setup,
  and same versus different room. Report accuracy on the pairs it answers and
  the "not sure" rate, in the format of section 5 of
  `docs/speaker-verification-hindi-english.md`.
- **Deepfake check:** every real clip should pass. Report the false-alarm rate
  by setup. Then make XTTS-v2 clones of consenting participants only
  (`scripts/make_fakes.py`) and report how many are caught.

## Consent template

> I agree to be recorded speaking for about 10 minutes for the B.S. Detector
> student project at <college>. The recordings will be used only to test the
> project's voice checks. They will not be published, shared outside the
> project team, or used to train AI models. They will be deleted by <date>, or
> earlier if I ask. [ ] I also agree that a synthetic copy of my voice may be
> made, only to test whether the project can detect it, and deleted on the same
> terms.
>
> Name, signature, date
