# Phase 7C — Dataset Metadata Inspection

**Status: metadata inspection only.** No audio was downloaded or processed,
no embeddings were generated, no ECAPA inference was run, no cosine
similarity/EER/FAR/FRR/threshold was calculated, and no code in
`src/speaker_verification/`, `scripts/`, or `tests/` was touched. The only
project modification is this document. Nothing has been committed.

This builds directly on `docs/dataset_audit_phase7b.md`. Where this
inspection **corrects** something reported in Phase 7B (this happened once,
for IIITH — see §3), that correction is called out explicitly rather than
silently overwritten, since it demonstrates exactly why direct metadata
inspection matters.

**What was actually inspected** (see §8 for full list): the official
`AI4Bharat/IndicSUPERB` GitHub README (which documents Kathbath in detail,
including per-language speaker/hour tables and the ASV task setup), the
`ai4bharat/Kathbath` Hugging Face dataset card, the HiACC Zenodo record
page and its Ulster University PURE repository listing (abstract-level
detail plus file/license metadata), and the full open-access ISCA-archive
paper for the IIITH Hindi-English Code-Mixed database (primary source,
not a secondary summary).

**What could not be reached in this environment**: the Kathbath ASV
`meta_data.zip` (hosted at `objectstore.e2enetworks.net`, blocked by this
sandbox's network allowlist — confirmed via an explicit `403
Forbidden` / `blocked-by-allowlist` response, the same restriction pattern
documented in `docs/model.md` for `huggingface.co`) and the full-text HiACC
PDF (hosted at `pure.ulster.ac.uk`; the same domain's HTML abstract page
was reachable, but the PDF itself returned no extractable content through
the available fetch tool, and the same host is also blocked at the sandbox
network layer per the same `blocked-by-allowlist` check). Both are
reported honestly below as **UNKNOWN / NOT DOCUMENTED FROM ACCESSIBLE
SOURCES**, per the rule not to infer unknown values, rather than guessed
at or bypassed.

---

## 1. Kathbath

### Metadata findings

Source: official `AI4Bharat/IndicSUPERB` GitHub README (the dataset's
official code/documentation repository — the highest-priority source type
available for this dataset) and the `ai4bharat/Kathbath` Hugging Face
dataset card.

| # | Question | Finding | Source |
|---|---|---|---|
| A | Total speakers/contributors | **1,218** contributors across all 12 languages, 203 districts | GitHub README overview; HF dataset card |
| B | Hindi-specific speaker count | **121** (58 male + 63 female) — read directly from the official per-language table | GitHub README, "Kathbath Dataset Details" table |
| C | Hindi-specific utterance count | UNKNOWN / NOT DOCUMENTED — the README table gives hours and vocabulary size (54k unique Hindi words), not an utterance/file count | GitHub README table |
| D | Hindi audio duration | **150.2 hours** | GitHub README table |
| E | Recordings/utterances per Hindi speaker | UNKNOWN exact count. A rough, explicitly-derived (not documented) figure: 150.2 h ÷ 121 speakers ≈ 1.24 h of audio per speaker on average, but this is a computed average across train/valid/test-known/test-unknown splits combined, not a stated "utterances per speaker" figure | Derived from D and B, not directly documented |
| F | Every Hindi speaker has multiple utterances? | Not explicitly stated as a number, but strongly implied: the dataset backs a **Speaker Identification (SID)** task (multi-class classification of speaker identity) and an **ASV** task, neither of which is meaningful with one utterance per speaker | GitHub README, "Speaker Identification" / "Automatic Speaker Verification" sections |
| G | Session independence (utterances genuinely independent recordings) | **UNKNOWN / NOT DOCUMENTED** — no session-ID field or recording-session description found in the accessible README/card | GitHub README, HF card |
| H | Speaker ID format | Encoded directly in the filename. Example from the README's folder-structure listing: `844424931537866-594-f.m4a` (numeric contributor id – numeric segment id – gender letter) | GitHub README, "Folder Structure of audios after extraction" |
| I | Speaker IDs consistent across recordings | Presumed yes (required for the SID/ASV tasks to function at all) but not stated in so many words | Inferred from task design, GitHub README |
| J | Session IDs / recording-session metadata | **UNKNOWN / NOT DOCUMENTED** — the visible filename pattern (id–id–gender) does not obviously encode a session field | GitHub README |
| K | Gender/demographic metadata | **Yes** — gender is in the filename itself, and the per-language table reports male/female speaker counts | GitHub README table and folder structure |
| L | Audio format | **`.m4a`** (compressed), NOT `.wav`/`.flac`. A conversion step (`utilities/structure.py`, provided in the IndicSUPERB repo) is required to get `.wav` | GitHub README, "Downloads" / "Data Preprocessing" |
| M | Sample rate | UNKNOWN / NOT DOCUMENTED in the README or dataset card | — |
| N | Existing train/validation/test splits | **Yes** — `train`, `valid`, `test_known` (testkn), `test_unknown` (testunk), plus separate "noisy" variants of the two test splits | GitHub README, "Downloads" |
| O | Existing ASV trial structure | **Yes, in principle** — a dedicated ASV metadata package (`meta_data.zip`) is published, containing per-language `valid_data.txt`, `test_known_data.txt`, `test_data.txt` files referenced directly in the ASV training/evaluation commands | GitHub README, "Automatic Speaker Verification" section |
| P | Official genuine/impostor trial pairs exist? | **Could not be confirmed** — `meta_data.zip` is the file that would contain the actual pairing/label format, but it could not be fetched in this environment (see note above; blocked by the sandbox's network allowlist). The ASV task description ("verifies whether the speakers of a pair of utterances match as a binary classification") strongly implies a genuine/impostor pair list exists inside it, but the exact format is **UNKNOWN / NOT DOCUMENTED FROM ACCESSIBLE SOURCES** | GitHub README description + failed fetch attempt |
| Q | Does the ASV benchmark use the same speaker identities we'd use? | Yes, by construction — the ASV task's training/eval commands point at the same `kb_data_clean_wav/<lang>` audio tree used for ASR/SID, so any Hindi-language subset we built from the same audio would draw on the same speaker pool the official benchmark uses | GitHub README |
| R | Exact metadata files needed to construct trials | `meta_data.zip` → `meta_data/hindi/{valid_data.txt, test_known_data.txt, test_data.txt}`, plus the raw Hindi `.m4a` audio tree (`data/hindi/<split>/<speaker>-<id>-<gender>.m4a`) | GitHub README |

### Speaker structure

121 Hindi speakers (58 M / 63 F), persistent speaker+gender-coded
filenames, split across 4 official data partitions (train/valid/testkn/
testunk). This is the only dataset in this inspection with an **explicit,
already-published ASV benchmark task** built on this exact data —
meaningfully stronger evidence of practical SV-suitability than a paper
just saying "speaker IDs exist."

### Hindi subset

150.2 hours, 121 speakers, 54k unique vocabulary words — real, substantial
scale, though notably smaller in speaker count than the Phase 7B
audit's aspirational 300–1,000-speaker target range.

### Trial feasibility

Genuine and impostor trial construction is **plausible and well-supported
by the existence of the official ASV task**, but I could not directly
confirm utterances-per-speaker or session independence (items E and G)
because the one file that would answer this precisely
(`meta_data.zip`) was not reachable from this environment. This is not a
"no" — it is a concrete, specific gap that direct access (from a machine
without this sandbox's network restrictions) would resolve quickly, since
the file is explicitly a small metadata package, not the audio corpus
itself.

### SV suitability

**POSSIBLY SUITABLE**, leaning toward READY — the existence of a
published, working ASV benchmark on this exact Hindi subset is strong
indirect evidence that genuine/impostor trials are constructible; the
classification is not "READY FOR SV" outright only because I could not
personally confirm the trial-pair file format and session independence
from documentation alone.

---

## 2. HiACC

### Metadata findings

Sources: the official Zenodo record (the dataset's official repository)
and the Ulster University PURE research-output page (which mirrors the
peer-reviewed *Data in Brief* journal article's abstract, bibliographic
metadata, and license).

| # | Question | Finding | Source |
|---|---|---|---|
| A | Total speakers | **UNKNOWN / NOT DOCUMENTED FROM ACCESSIBLE SOURCES** — neither the Zenodo record nor the abstract states a speaker count | Zenodo record; Ulster PURE abstract |
| B | Adult speakers | UNKNOWN (same reason) | — |
| C | Child speakers | UNKNOWN (same reason) | — |
| D | Number of unique speaker IDs | UNKNOWN | — |
| E | Number of utterances | **3,318 adult + 1,858 child audio segments** (5,176 total), 5.24 hours combined | Zenodo record description; Ulster PURE abstract (identical figures in both) |
| F | Utterances per speaker | UNKNOWN — cannot be computed without a speaker count | — |
| G | Every speaker has multiple utterances? | UNKNOWN | — |
| H | Multiple sessions? | UNKNOWN | — |
| I | Session IDs? | UNKNOWN | — |
| J | Genuinely independent same-speaker recordings? | UNKNOWN | — |
| K | Speaker ID structure | UNKNOWN | — |
| L | Speaker IDs persist across all recordings? | UNKNOWN | — |
| M | Contains Hindi and English/code-switched speech? | **Yes**, explicitly — this is the corpus's entire purpose (Hinglish code-switching) | Zenodo/abstract description |
| N | Language/code-switching labels exist? | **Yes** — "detailed annotations and code-switching tags" are explicitly mentioned | Ulster PURE abstract |
| O | Audio format | UNKNOWN / NOT DOCUMENTED FROM ACCESSIBLE SOURCES | — |
| P | Sample rate | UNKNOWN | — |
| Q | Genuine speaker trials constructible? | **UNKNOWN — cannot be determined without the information in rows A–L** | — |
| R | Impostor trials constructible? | UNKNOWN, for the same reason (would need at least a speaker count) | — |
| S | Official SV protocol? | **No** — the corpus's own baseline experiments (in the abstract) are ASR word-error-rate experiments, not speaker verification | Ulster PURE abstract |
| T | Exact metadata/file structure needed | Zenodo currently exposes exactly **one** downloadable file, `Corpus.zip` (531.6 MB, confirmed exact size from the Zenodo file listing) — there is no separate small metadata/CSV file to preview; the per-speaker/session structure is described only in the paper's full body text | Zenodo record file listing |

**On the "1 recording = NOT SUITABLE" test specifically**: I want to be
precise about what I do and don't know here. I did **not** find evidence
that each HiACC speaker has only one recording — I found **no speaker
count or session information at all** in the sources I could reach. The
paper's abstract explicitly distinguishes "read and spontaneous speech"
and describes three distinct elicitation methods (daily-life
questionnaires, story reading, image-based questions), which is at least
suggestive of multiple recording contexts per participant, but this is
not confirmed. Marking this **NOT SUITABLE** would be guessing; the honest
classification is **UNKNOWN**.

The full peer-reviewed paper (open access, **CC BY 4.0**, confirmed via
the Ulster PURE page's licence tag) is hosted at
`https://pure.ulster.ac.uk/files/224502270/1-s2.0-S2352340925006109-main.pdf`
and would very likely answer items A–L directly — it just was not
retrievable by the fetch tooling available in this environment (empty
response) and the host is also blocked at the sandbox network-policy
layer. **This is the single most valuable next inspection step for
Hinglish** (see §7).

### Speaker structure

Not determinable from accessible sources.

### Trial feasibility

Not determinable from accessible sources — genuinely open question, not a
negative finding.

### SV suitability

**UNKNOWN.**

---

## 3. IIITH Hindi-English Code-Mixed Speech Database

### Metadata findings

Source: the actual dataset paper, "*Development of IIITH Hindi-English
Code Mixed Speech Database*" (Rambabu & Gangashetty, SLTU 2018), fetched
in full from the official ISCA Archive (open access, no paywall) — a
primary source, not a secondary summary.

**Correction to Phase 7B**: the earlier audit reported "78 speakers,
~1 minute, apparently single session" for this dataset, sourced from a
search-result snippet rather than the paper itself. Having now read the
actual paper, both parts of that figure need correcting:

- **Total speakers is 142, not 78** (110 in the training split + 32 in
  the test split, each split gender-balanced).
- **Each speaker recorded in 3 separate sessions** (not one) — the paper
  states explicitly: *"Each speaker was recorded in three sessions... with
  a 5 minutes vocal rest between the sessions."* The "~1 minute" figure
  from Phase 7B is roughly right only as a **per-session** duration
  estimate (see table row D), not as the whole per-speaker recording.

This is exactly the kind of correction direct metadata inspection is for
— the Phase 7B secondary-source figure understated both the speaker count
and (more importantly for SV suitability) the number of independent
recording sessions per speaker.

| # | Question | Finding | Source |
|---|---|---|---|
| A | Total speakers | **142** (71 male, 71 female) | Paper §4.1, §5 |
| B | Speaker IDs | Yes — each wave file's unique ID encodes speaker name + gender + file id | Paper §4.3 |
| C | Recordings per speaker | **~62 unique sentences per speaker** (6,820 unique training utterances ÷ 110 training speakers ≈ 62; 1,984 unique test utterances ÷ 32 test speakers ≈ 62), split across 3 recording sessions (~20 sentences/session, per §4.2: "Each session contains 20 sentences to speak") | Paper §4.2, §5.1, Table 1 |
| D | Duration per recording | Not stated per-utterance. Corpus-level: ~8 hours training / ~3 hours test ≈ **11 hours total** across 142 speakers ⇒ ≈4.6 minutes of audio per speaker in total, ⇒ roughly ~1.5 minutes per session (3 sessions/speaker) — in the right range of Phase 7B's "~1 minute" figure, but per SESSION, not per speaker overall | Paper §5.1, §7 (Summary) |
| E | Each speaker has multiple recordings? | **Yes — confirmed explicitly**, 3 sessions per speaker | Paper §4.2 |
| F | Separate sessions? | **Yes, explicitly stated**, with a 5-minute vocal-rest break enforced between sessions | Paper §4.2 |
| G | Different recording conditions? | **Partially independent**: sessions are temporally separated (with a rest break) but appear to occur on the same day, same sound-proof studio, same microphone/equipment/distance instructions (paper specifies "maintain a distance of 4-6 inches from the microphone" as a constant instruction) — this is real temporal separation, not full multi-day/multi-device independence like VoxCeleb's different-YouTube-video structure | Paper §4.2 |
| H | Is the Phase 7B "~1-minute-per-speaker, single session" description accurate? | **No, partially inaccurate** — see correction note above. ~1 minute is approximately right as a *per-session* estimate; "single session" is incorrect, there are 3 | This inspection, §3 correction note |
| I | Separate utterances within each recording? | Yes — each session comprises ~20 distinct sentence-level utterances, each apparently saved as its own file (the corpus is described in terms of "unique utterances," e.g. 6,820 + 1,984 = 8,804 total, matching the 8,804 unique sentences in the text corpus) | Paper §3.2, Table 1, §5.1 |
| J | Can utterances reasonably be considered independent? | Within a session: they're different sentences but recorded consecutively, same take. Across sessions: yes, more genuinely independent (different time, vocal-rest break) | Paper §4.2 |
| K | Genuine trials constructible? | **Yes, plausible** — pairing an utterance from session 1 against an utterance from session 2 (or 3) for the same speaker gives a same-speaker, cross-session pair, which is a reasonable genuine-trial definition | Derived from A–J |
| L | Impostor trials constructible? | **Yes** — 142 distinct real speakers, trivially pairable across speakers | Paper §4.1 |
| M | Audio format | WAV ("wave files"), mono | Paper §4.2, §4.3 |
| N | Sample rate | Recorded at 48 kHz / 24-bit PCM mono; **released/segmented version is downsampled to 16 kHz / 16-bit PCM mono** — this matches this project's existing `audio.py` target sample rate (`TARGET_SAMPLE_RATE = 16_000`) natively, unlike Kathbath's `.m4a` format, which would need a conversion step first | Paper §4.3 |
| O | Metadata structure | Filename encodes speaker name + gender + file id (exact session-number encoding in the filename is **not confirmed** by the paper text — this specific sub-question is **UNKNOWN**, would need to see an actual file listing); a per-state speaker-count table is given (Uttar Pradesh 30, Rajasthan 21, Bihar 18, Madhya Pradesh 18, Delhi 9, Haryana 8, Maharashtra 7, and 12 other states with 1–6 speakers each) | Paper §4.1, Figure 1 |
| P | Access requirements | **UNKNOWN / NOT DOCUMENTED** — the paper gives no dataset URL, no download link, and no explicit availability statement. It does give author contact emails (`rambabu.b@research.iiit.ac.in`, `svg@iiit.ac.in`, IIIT Hyderabad Speech Processing Laboratory). This specific corpus is **not** listed on IIIT-H's general "IIIT-H Indic Speech Databases" public page (which only lists the monolingual Bengali/Hindi/Kannada/Malayalam/Marathi/Tamil/Telugu TTS databases) — access likely requires directly emailing the authors, unconfirmed | Paper (no URL given); http://speech.iiit.ac.in/index.php/research-svl/69.html (does not list this corpus) |
| Q | License, if visible | **Not stated in the paper** — LICENSE UNCLEAR / NOT DOCUMENTED | Paper (no license section) |

### Speaker structure

142 speakers (71 M / 71 F), each with 3 time-separated recording
sessions and ~62 short utterances total, drawn from 19 Indian states —
this is the richest, most concretely-documented Hinglish candidate found
in either audit.

### Trial feasibility

Genuine trials (same speaker, different session) and impostor trials
(different speakers) both appear constructible from the documented
structure. The main open technical question is whether the actual
released filenames let you tell which session an utterance came from
(needed to deliberately pick a cross-session pair rather than an
accidental same-session pair) — this needs a real file listing to
confirm, not just the paper text.

### SV suitability

**POSSIBLY SUITABLE** — meaningfully upgraded from Phase 7B's implicit
pessimism (which was based on an inaccurate secondary-source figure). Not
labeled READY FOR SV because (a) session-level file-naming isn't
confirmed, (b) access/license terms are completely undocumented, and (c)
within-session utterances share equipment/environment/day, which is a
weaker independence guarantee than VoxCeleb-style different-video pairs.

---

## 4. Side-by-side comparison

| | Kathbath (Hindi) | HiACC (Hinglish) | IIITH Hindi-English (Hinglish) |
|---|---|---|---|
| Speakers (relevant subset) | 121 (Hindi) | UNKNOWN | 142 |
| Recordings/utterances | UNKNOWN exact count; 150.2 h Hindi audio | 5,176 segments (3,318 adult + 1,858 child), 5.24 h | ~8,804 utterances (62/speaker × 142), ~11 h |
| Recordings per speaker | UNKNOWN exact (avg ≈1.24 h/speaker, not an utterance count) | UNKNOWN | ~62 utterances across 3 sessions |
| Session independence | UNKNOWN / NOT DOCUMENTED | UNKNOWN | Partial — 3 sessions, same day, vocal-rest breaks between them |
| Speaker IDs | Yes, filename-encoded (id-id-gender) | UNKNOWN | Yes, filename-encoded (name+gender+file id) |
| Hindi/Hinglish relevance | Hindi (monolingual) | Hinglish (explicit code-switching) | Hindi-English code-mixed (explicit) |
| Genuine trials | Plausible (backed by a real ASV benchmark) | UNKNOWN | Plausible (documented multi-session structure) |
| Impostor trials | Yes (121 speakers) | UNKNOWN (speaker count unknown) | Yes (142 speakers) |
| Speaker-disjoint split | Yes — dataset ships train/valid/testkn/testunk already, and our own `split_trials_by_speaker` would also work on any subset | Cannot assess | Yes — paper's own train (110)/test (32) split is speaker-disjoint by design |
| License/access | CC0 per explicit text (badge on the HF page inconsistently shows "cc-by-4.0" — **flagging this discrepancy**, treat as needing re-verification); gated HF download requiring contact-info agreement | **CC BY 4.0**, confirmed on Ulster PURE; open Zenodo download, no gate | **UNKNOWN / NOT DOCUMENTED** — no license stated in the paper, no public download link found |
| Storage (see §8) | Full Hindi portion likely several GB (proportional estimate); a curated subset much smaller | 531.6 MB for the ENTIRE corpus (confirmed exact) | ~1.3 GB estimated for the entire corpus (16 kHz/16-bit mono WAV, 11 hours) |
| SV suitability (this inspection) | **POSSIBLY SUITABLE**, leaning READY | **UNKNOWN** | **POSSIBLY SUITABLE** |

---

## 5. Final Hinglish decision

**HiACC: UNKNOWN.** Not enough accessible information to classify it
either way — the speaker count and session structure simply weren't in
any source I could reach. This is very likely resolvable (the full paper
is openly licensed and almost certainly contains this information) but
was not resolvable *in this environment* today.

**IIITH Hindi-English Code-Mixed: POSSIBLY SUITABLE**, and the stronger
lead of the two right now — 142 documented speakers, an explicit
3-session recording protocol with enforced breaks, a paper-defined
speaker-disjoint train/test split, and audio already at this project's
target 16 kHz mono format. It falls short of READY FOR SV only because
session-level file naming and licensing/access terms are unconfirmed.

**Neither dataset is being selected for calibration in this phase — that
decision is explicitly deferred, per your instructions.** If forced to
prioritize which one to chase down first, I would pursue **IIITH first**
(read the paper's cited contact emails or its parent lab page; the
structure is already well-documented and just needs an access channel),
while independently trying to obtain the HiACC full-text PDF directly
(rather than through this sandbox) to resolve its remaining unknowns,
since it's freely downloadable once reachable.

## 6. Recommended final dataset strategy

- **English: VoxCeleb1**, unchanged from Phase 7B — nothing in this
  inspection touched English datasets.
- **Hindi: Kathbath**, reinforced by this inspection. 121 real,
  documented Hindi speakers with an existing, working ASV benchmark on
  the exact same data is a strong basis; the one remaining gap
  (utterances-per-speaker, session independence) needs the
  `meta_data.zip` file, which requires access from a machine without this
  sandbox's network restrictions.
- **Hinglish: still not selected.** Recommend obtaining direct access to
  IIITH Hindi-English Code-Mixed (via the authors' contact emails in the
  paper) as the next concrete step, since it is now the better-understood
  of the two candidates; keep HiACC as a strong second option pending
  reading its full (openly licensed) paper text.

**Do not implement any of this yet** — per your instructions, this is a
recommendation only.

## 7. Remaining unknowns (require actual dataset/file access)

- **Kathbath**: exact Hindi utterance count, utterances-per-speaker,
  session independence, sample rate, and the precise genuine/impostor
  pairing format — all contained in `meta_data.zip`
  (`https://objectstore.e2enetworks.net/indic-superb/meta_data.zip`),
  which this sandbox could not reach (network-allowlist blocked). This is
  a small metadata package (not audio), explicitly allowed to inspect per
  your rules — just not reachable from here.
- **Kathbath**: the license badge inconsistency (CC0 in text vs.
  "cc-by-4.0" badge on the HF page) should be resolved by reading the
  dataset's actual license file/agreement at download time.
- **HiACC**: total speaker count, adult/child breakdown, session
  structure, speaker-ID format, audio format/sample rate — all likely
  answered in the full paper body
  (`https://pure.ulster.ac.uk/files/224502270/1-s2.0-S2352340925006109-main.pdf`,
  confirmed CC BY 4.0, 2.51 MB), which returned no extractable content
  through the fetch tool available here and whose host is also blocked at
  the sandbox network-policy layer.
- **IIITH**: whether session identifiers are recoverable from the
  released filenames; the corpus's actual license and access/download
  procedure (not stated in the paper; would require contacting the
  authors directly).
- For all three: exact per-file audio sizes (to firm up the storage
  estimates in §8 below beyond the documented totals).

## 8. Sources

Actual metadata / official repositories consulted directly (not just
search snippets):

- [AI4Bharat/IndicSUPERB — official GitHub repository](https://github.com/AI4Bharat/IndicSUPERB) (fetched in full: Kathbath per-language table, folder structure, ASV/SID/ASR task descriptions, download links, license section)
- [ai4bharat/Kathbath — Hugging Face dataset card](https://huggingface.co/datasets/ai4bharat/Kathbath) (fetched in full: languages, licensing text, total file size, access-gate notice)
- [HiACC: Hinglish Adult & Children Code-switched corpus — Zenodo record](https://zenodo.org/records/15551669) (fetched in full: file listing/size, license badge, description, dates)
- [HiACC: Hinglish adult & children code-switched corpus — Ulster University PURE](https://pure.ulster.ac.uk/en/publications/hiacc-hinglish-adult-amp-children-code-switched-corpus/) (fetched in full: abstract, DOI, license tag on the PDF link, journal metadata)
- [Development of IIITH Hindi-English Code Mixed Speech Database — ISCA Archive (full paper PDF)](https://www.isca-archive.org/sltu_2018/rambabu18_sltu.pdf) (fetched and read in full — primary source)
- [IIIT-H Indic Speech Databases — official lab page](http://speech.iiit.ac.in/index.php/research-svl/69.html) (fetched in full — confirms this page does NOT list the Hindi-English code-mixed corpus)

Attempted but not accessible from this environment (reported above as
UNKNOWN rather than guessed):

- `https://objectstore.e2enetworks.net/indic-superb/meta_data.zip` (Kathbath ASV metadata — blocked by sandbox network allowlist, confirmed via HTTP 403 / `blocked-by-allowlist`)
- `https://pure.ulster.ac.uk/files/224502270/1-s2.0-S2352340925006109-main.pdf` (HiACC full paper — no extractable content returned; host also blocked by the sandbox network allowlist)
- `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12329218/` (HiACC PMC mirror — blocked by an anti-bot/reCAPTCHA interstitial)
- `https://www.sciencedirect.com/science/article/pii/S2352340925006109` and its DOI-resolver redirect target (HiACC ScienceDirect page — no extractable content returned)
- `https://datasets-server.huggingface.co/info?dataset=ai4bharat/Kathbath` and `.../splits?dataset=...` (Hugging Face datasets-server API — no content returned, consistent with this project's documented `huggingface.co` network restriction)
- `https://huggingface.co/datasets/ai4bharat/Kathbath/raw/main/README.md` (no content returned; the same information was successfully obtained instead from the mirrored GitHub README and the rendered HF dataset-card page)

Carried over from Phase 7B (not re-fetched in this phase, referenced for
context only): `docs/dataset_audit_phase7b.md` and its own source list.
