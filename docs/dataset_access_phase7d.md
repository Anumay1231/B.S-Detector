# Phase 7D — Dataset Access & Final Verification

**Status: access verification and small-metadata acquisition only.** No
audio was downloaded or processed, no embeddings were generated, no ECAPA
inference was run, no similarity/EER/FAR/FRR/threshold was calculated, and
no code in `src/speaker_verification/`, `scripts/`, or `tests/` was
touched. The only project modification is this document. Nothing has been
committed.

**Headline finding: this sandbox's own network policy blocked every
attempted metadata/access check this phase actually needed to resolve.**
Every URL below that returned `403 Forbidden` / `X-Proxy-Error:
blocked-by-allowlist` was blocked by this environment, not by the dataset
providers — this is the same restriction already documented for
`huggingface.co` and `download.pytorch.org` since Phase 4
(`docs/model.md`). None of these datasets appear to require anything
unusual; they simply cannot be reached from here. This phase therefore
ends with a **clear list of manual steps for you**, not a dead end.

## 1. Kathbath

**Access status: BLOCKED (sandbox network policy), not dataset-side.**

I attempted to fetch `meta_data.zip` — the ASV trial-metadata package
identified in Phase 7C — via both the fetch tool and a direct HTTP
request:

```
HEAD https://objectstore.e2enetworks.net/indic-superb/meta_data.zip
→ HTTP/1.1 403 Forbidden
  X-Proxy-Error: blocked-by-allowlist

GET  https://objectstore.e2enetworks.net/indic-superb/meta_data.zip
→ connection failed (same allowlist block)
```

**Metadata package status: NOT ACQUIRED.** Per your instruction ("If
access fails: STOP that subtask and report exactly why"), I'm stopping
here rather than trying an unofficial mirror or working around the block.

**Exact trial structure / exact speaker-utterance information**: unchanged
from Phase 7C — still confirmed only at the level of the official
`AI4Bharat/IndicSUPERB` GitHub README (re-verified reachable this phase,
same source as before):

- 121 Hindi speakers (58 male, 63 female), 150.2 hours of Hindi audio
- ASV task metadata files are named `valid_data.txt`,
  `test_known_data.txt`, `test_data.txt` per language (referenced directly
  in the README's ASV training/eval commands)
- Audio format `.m4a`, requiring conversion to `.wav` via a provided
  utility script before use with this project's `audio.py` (which only
  accepts `.wav`/`.flac`)

Items A (exact Hindi utterance count), C (multi-recording independence), D
(session info), E (exact trial format), F (exact label encoding), G
(exact split structure beyond the 4 named splits), H (sample rate), and I
(exact filename/path format beyond the `id-id-gender.m4a` pattern already
seen) remain **UNKNOWN / NOT DOCUMENTED FROM ACCESSIBLE SOURCES** — the
one file that would answer all of them (`meta_data.zip`) could not be
opened from this environment.

Item J (whether the official ASV trials can be directly reused by our
pipeline): **plausible but unconfirmed** — our `Trial` dataclass
(`reference_id`, `test_id`, `label`, `reference_speaker`, `test_speaker`)
is generic enough to represent whatever `meta_data.zip` turns out to
contain, the same way `voxceleb.py` maps VoxCeleb's own format onto it.
Whether that requires new adapter logic or a trivial reformat can only be
answered once the file's actual contents are visible.

**License**: unchanged from Phase 7C — CC0 stated explicitly in the
IndicSUPERB README text; the Hugging Face dataset page's license *badge*
inconsistently shows "cc-by-4.0" (this discrepancy is still unresolved and
should be checked directly against Hugging Face's dataset-access terms,
which also could not be reached this session — see VoxCeleb section for
the same underlying block pattern).

**Remaining issues**: `meta_data.zip` needs to be downloaded from a
machine without this sandbox's restrictions (see §8, manual action #1).

## 2. IIITH Hindi-English Code-Mixed Database

**Access status: NOT PUBLICLY DOWNLOADABLE, as far as could be
determined.**

- **A/B (where obtained, official URL)**: No dataset URL, download link,
  or availability statement appears anywhere in the paper itself. I
  re-checked IIIT-H's general "IIIT-H Indic Speech Databases" page
  (`http://speech.iiit.ac.in/index.php/research-svl/69.html`) — it lists
  only the monolingual Bengali/Hindi/Kannada/Malayalam/Marathi/Tamil/
  Telugu TTS databases, **not** this Hindi-English code-mixed corpus.
  A broader search for an official download page or a dedicated IIIT-H
  Speech Processing Laboratory datasets listing could not be completed
  this session (the search tool hit its session rate limit partway
  through this phase) — this specific check is incomplete, not negative.
- **C/D/E (session IDs in filenames, exact convention, session
  recoverability)**: the paper states filenames encode "speaker name,
  gender of the speaker and wave file [id]" — it does **not** state that a
  session number is part of that encoding. I have no actual file listing
  to check this against.
- **F/G (public downloadability / official access procedure)**: not
  publicly downloadable as far as I can tell; the only access path
  visible is the two author emails printed in the paper.
- **H/I/J (license, research use, SV-research use)**: **not stated
  anywhere in the paper** — LICENSE UNCLEAR / NOT DOCUMENTED.

### Applying Phase 7D's explicit file-structure rule (§3 of your request)

Your instructions are explicit: *"If session identity cannot be recovered
from the released filenames/metadata: mark NOT READY FOR IMPLEMENTATION.
Do not infer session identity from file ordering."*

I cannot currently confirm session identity is recoverable from the
released filenames — the paper's prose doesn't say either way, and I have
no actual directory listing to inspect. Per your rule, the correct
classification is:

**IIITH: NOT READY FOR IMPLEMENTATION.**

This is a stricter conclusion than Phase 7C's "POSSIBLY SUITABLE" — that
earlier label was about structural plausibility; this phase's rule is
specifically about whether we can *prove* session recoverability today,
and we cannot.

### Manual contact information (per your instructions — I have NOT contacted anyone)

If you want to pursue this dataset, here is exactly what I'd suggest
sending, for you to review and send yourself:

- **To**: `rambabu.b@research.iiit.ac.in` and `svg@iiit.ac.in` (the
  paper's listed authors, IIIT Hyderabad Speech Processing Laboratory)
- **Reason to state**: academic/research interest in using the IIITH
  Hindi-English Code-Mixed (IIITH-HE-CM) speech database described in
  their SLTU 2018 paper ("Development of IIITH Hindi-English Code Mixed
  Speech Database") for speaker-verification research, as part of a
  bilingual (Hindi/English) speaker-verification project.
- **Exact dataset/version to request**: the IIITH-HE-CM corpus described
  in that paper — 142 speakers, ~8,804 utterances, ~11 hours, 16 kHz/16-bit
  mono WAV.
- **Exact information to ask for**:
  1. Whether the corpus (or a research-use subset of it) is available for
     download, and under what license/terms.
  2. Whether session number (1, 2, or 3) is encoded in the released
     filenames or in a separate metadata/manifest file.
  3. Whether speaker-verification research use (not just ASR) is
     permitted under their terms.
  4. Whether redistribution or storage of derived embeddings (not raw
     audio) is permitted.

## 3. VoxCeleb1

**Access status: could not be freshly re-verified from this sandbox.**

```
HEAD https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
→ HTTP/1.1 403 Forbidden
  X-Proxy-Error: blocked-by-allowlist
```

This is the same sandbox-level network block as Kathbath's metadata file
— the official VGG VoxCeleb page itself is unreachable from here. I am
**not** substituting an unofficial mirror to work around this, per your
instructions.

- **Current official access procedure / licensing terms**: unchanged from
  Phase 7B's findings (access via VGG's own request process; terms
  reported inconsistently across secondary sources at that time, flagged
  "LICENSE PARTIALLY UNCLEAR, verify at request time"). I could not
  independently re-confirm the *current* wording this session — that
  still needs to happen on the official page directly, from a machine
  that can reach it.
- **Official trial-list availability / exact format**: VoxCeleb1's
  verification trial lists (the base list plus VoxCeleb1-E and
  VoxCeleb1-H, and their "cleaned" variants) are a long-standing, stable,
  widely-documented format in the speaker-verification research
  community: one trial per line, `<label> <reference_path> <test_path>`,
  where label is `1` (genuine) or `0` (impostor). This is general
  knowledge from the broader research literature (already used to design
  `voxceleb.py` in Phase 7A), not something re-confirmed live this
  session — if VGG has changed the format, that would only surface once
  the file is actually obtained.
- **Does the existing `voxceleb.py` adapter match the current official
  structure?** It was built against, and its tests exercise, exactly that
  stable 3-field format. I did not change it this phase (not asked to,
  and not appropriate without first re-confirming the live file). If the
  official format has changed in some way I'm not aware of, that would
  only be discovered once real trial-list text is obtained and run
  through `parse_trial_list()` — which would surface as a clear
  `TrialListParseError` rather than silently mis-parsing, since the
  parser validates field count and label values strictly.
- **Small controlled subset**: yes, unaffected by any of this —
  `scripts/build_trials.py --max-genuine/--max-impostor/--seed` already
  supports exactly this, and doesn't require the full dataset on disk
  first (just the trial-list text file plus whichever audio files are
  actually referenced by the sampled subset).

## 4. HiACC

**Still worth pursuing?** Yes, as backup — nothing here changes Phase 7C's
assessment. **Still unknown**: speaker count, adult/child breakdown,
session structure, speaker-ID format, audio format/sample rate — all
likely in the full paper body, which remains unreachable from this
sandbox:

```
GET https://pure.ulster.ac.uk/files/224502270/1-s2.0-S2352340925006109-main.pdf
→ no content returned (host also blocked at the sandbox network-policy layer,
   confirmed for the sibling VoxCeleb/Kathbath domains this same session)
```

The PDF is confirmed CC BY 4.0 and only 2.51 MB — trivial to open from any
unrestricted machine or browser.

## 5. Final dataset decision

Using your requested classification (READY / READY WITH ONE KNOWN MANUAL
STEP / NOT READY / REJECTED):

- **VoxCeleb1: READY WITH ONE KNOWN MANUAL STEP** — the adapter is already
  built and tested against the documented format; the one step is you
  completing VGG's access request and obtaining a trial-list file from a
  machine that can reach `robots.ox.ac.uk`.
- **Kathbath: READY WITH ONE KNOWN MANUAL STEP** — 121 real, documented
  Hindi speakers and an existing ASV benchmark are a solid foundation; the
  one step is fetching and reading `meta_data.zip` from an unrestricted
  machine (small file, not the audio).
- **IIITH: NOT READY** — per the explicit rule in §3 of your request:
  session-identity recoverability from filenames is unconfirmed, and
  access/license terms are completely undocumented. Two open blockers,
  not one.
- **HiACC: BACKUP, effectively NOT READY** — speaker/session structure
  still entirely unknown; would need the same kind of direct-access step
  as the others once someone can actually open the PDF.

**English and Hindi are each one concrete, well-defined manual step away
from real implementation. Hinglish is not — pursuing IIITH further (via
direct author contact) or HiACC further (via someone actually reading the
open-access PDF) are both still open, parallel paths.**

## 6. Recommended first experiment

Given the above, only English and Hindi can be sized concretely right
now:

- **English (VoxCeleb1)**: 300–500 speakers, ~1,000 genuine + ~1,000
  impostor trials — matches this project's existing Phase 7H precedent,
  well within VoxCeleb1's 1,251 speakers. Contingent on completing the
  access request.
- **Hindi (Kathbath)**: up to 121 Hindi speakers (a hard ceiling from the
  actual dataset — below the original 300–1,000 aspirational range, and
  that's a real, documented constraint, not an implementation choice).
  Genuine/impostor trial counts cannot be finalized until `meta_data.zip`
  is read; a reasonable provisional target once that's available would be
  "as many genuine/impostor pairs as the official ASV valid/test splits
  actually support" rather than a pre-committed number, per your
  instruction to prioritize reliable trials over arbitrary size.
- **Hinglish**: no experiment sized yet — not ready.
- **Calibration split**: reuse `split_trials_by_speaker()` unchanged, per
  your instruction. Recommend **option E (both language-specific and
  global, for comparison)** as the long-term design once at least two
  languages are actually available — calibrate/evaluate English and Hindi
  separately (since cross-language genuine trials are never valid, per
  the standing rule), and only *additionally* look at a pooled
  presentation for descriptive comparison, never for threshold selection
  itself. Practically, right now that means English-specific and
  Hindi-specific tracks; the "global comparison" side of option E has to
  wait for Hinglish (or is meaningless with only 2 languages pooled in a
  way that still respects the no-cross-language-genuine-trials rule).

## 7. Storage estimate

| Dataset | Full dataset size | Required subset size | Files | Duration | Expected storage |
|---|---|---|---|---|---|
| VoxCeleb1 | ~30–40 GB (commonly cited, not re-verified this session) | A few thousand short utterances touched by a ~2,000-trial subset | Low thousands of files | Minutes total across sampled utterances | Low single-digit GB |
| Kathbath (Hindi) | Hindi portion ≈150.2 h of the full 1,684 h / 174 GB dataset (≈8.9% share ⇒ rough proportional estimate ~15 GB for the ENTIRE Hindi portion, all speakers/splits) | A curated subset touching a handful of utterances per speaker across 121 speakers | Low thousands of `.m4a` files (compressed) | Tens of minutes to low hours, depending on utterances/speaker (still unconfirmed) | Likely well under 1 GB for a curated subset, pending confirmation |
| IIITH | ~11 h, 142 speakers, 16 kHz/16-bit mono WAV | Not applicable — NOT READY | ~8,804 files if the full corpus were used | ~11 h | ~1.3 GB estimated for the ENTIRE corpus (uncompressed 16 kHz mono WAV math), if it ever becomes usable |
| HiACC | 531.6 MB (confirmed exact, Zenodo file listing) | Not applicable — NOT READY | 5,176 segments | 5.24 h | Whole corpus already fits comfortably in the storage budget, if it ever becomes usable |

**Total for a realistic first English+Hindi experiment: comfortably within
the 5–10 GB target**, likely closer to 1–3 GB, without needing to
artificially truncate anything — the actual constraint on Hindi scale is
Kathbath's real speaker count (121), not a storage limit.

## 8. Manual actions required from me

These require a machine without this sandbox's network restrictions
(basically any normal internet connection — the block is specific to this
sandboxed environment, not to you):

1. **Download `https://objectstore.e2enetworks.net/indic-superb/meta_data.zip`**
   (small — this is a metadata/trial-list package, not audio; exact size
   unconfirmed since I couldn't reach it, but text trial lists for 12
   languages should be well under 100 MB). Unzip it and share back a few
   example lines from `meta_data/hindi/valid_data.txt`,
   `test_known_data.txt`, and `test_data.txt` so I can confirm the exact
   genuine/impostor pairing format before any adapter is designed.
2. **Open `https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html`**
   directly and confirm the current access-request process and license
   wording, since I could not reach this page from here this session.
3. **If you want to pursue IIITH**: send the email described in §2 to
   `rambabu.b@research.iiit.ac.in` and `svg@iiit.ac.in` yourself (I have
   not, and will not, contact them automatically).
4. **If you want to pursue HiACC instead/also**: open
   `https://pure.ulster.ac.uk/files/224502270/1-s2.0-S2352340925006109-main.pdf`
   (free, CC BY 4.0, 2.51 MB) and check whether it states a speaker count
   and session structure — that single check would resolve HiACC's
   remaining unknowns.

None of these require downloading a full multi-GB dataset — all four are
either a small metadata file or reading a page/PDF that happens to be
unreachable specifically from this sandbox.

## Sources

Re-verified as reachable this session (fetched in full):

- [AI4Bharat/IndicSUPERB — official GitHub repository](https://github.com/AI4Bharat/IndicSUPERB)
- [IIIT-H Indic Speech Databases — official lab page](http://speech.iiit.ac.in/index.php/research-svl/69.html)

Confirmed blocked by this sandbox's network policy this session (HTTP 403
/ `blocked-by-allowlist`, verified via direct request):

- `https://objectstore.e2enetworks.net/indic-superb/meta_data.zip` (Kathbath ASV metadata)
- `https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html` (official VoxCeleb1 page)
- `https://api.github.com/repos/AI4Bharat/IndicSUPERB/contents/file` (GitHub API path)
- `https://pure.ulster.ac.uk/files/224502270/1-s2.0-S2352340925006109-main.pdf` (HiACC full paper PDF)

Carried over from Phase 7B/7C (not re-fetched this phase): see the source
lists in `docs/dataset_audit_phase7b.md` and
`docs/dataset_metadata_inspection_phase7c.md`.

**Note on incomplete coverage this phase**: a broader web search for an
official IIITH-HE-CM download page was cut short by a search-tool session
limit partway through this phase. This specific check (§2, items A/B) is
incomplete rather than concluded — it's a reasonable next step for a
follow-up session, not a hard blocker reflected in the NOT READY verdict,
which is driven primarily by the unconfirmed session-recoverability
question.
