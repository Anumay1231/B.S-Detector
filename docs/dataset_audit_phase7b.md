# Phase 7B — Dataset Discovery, Audit & Selection (English / Hindi / Hinglish)

**Status: research and recommendation only. No datasets downloaded, no audio
processed, no embeddings generated, no calibration run. Nothing in
`src/speaker_verification/`, `scripts/`, or `tests/` was changed to produce
this document.** This file is not yet committed to Git — it is a draft for
review before any Phase 8 work begins.

This document audits candidate public datasets for building a combined
English + Hindi + Hinglish speaker-verification calibration/evaluation
corpus on top of the existing, unchanged pipeline (`audio.py` → `encoder.py`
→ `similarity.py` → `calibration.py`, plus the Phase 7 VoxCeleb pipeline in
`src/speaker_verification/datasets/`).

All statistics below come from the cited sources (official dataset pages,
papers, Hugging Face dataset cards, LDC/GitHub catalog pages). Anything I
could not verify from those sources is explicitly marked **UNKNOWN / NOT
DOCUMENTED** rather than estimated.

---

## SECTION 1 — Best English datasets

**VoxCeleb1** and **VoxCeleb2** remain the strongest English candidates —
this is not an automatic choice, it's a comparison result (see Section 4):
they are the only English candidates with an *official, published
verification trial-list protocol* (the exact structure this project's
Phase 7 pipeline already consumes), thousands of speakers, and multiple
independent recordings per speaker from real-world conditions.

**LibriSpeech** is a strong second candidate: much easier licensing (CC BY
4.0, no request form) and 2,000+ speakers, but it was built for ASR, not
speaker verification — any trial list has to be a community-constructed
one, not an official protocol, and its read-audiobook conditions are far
cleaner/less "in the wild" than VoxCeleb or than this project's real target
audio.

**SITW** (Speakers in the Wild) is a legitimate, purpose-built SV dataset
with genuinely challenging real-world conditions, but it is small (299
speakers) and requires a request to SRI International — useful as a
supplementary hard-evaluation set, not as the primary English source.

**Recommendation for English: VoxCeleb1 as primary (Phase 7 already
supports it), LibriSpeech as an optional easy-license supplement if a
second, cleaner-condition English source is wanted.**

## SECTION 2 — Best Hindi/Indian datasets

**Kathbath** (AI4Bharat, part of the IndicSUPERB benchmark) is the
strongest Hindi/Indian candidate found: it is the *only* Hindi-relevant
dataset in this audit that is explicitly documented as being used for an
**Automatic Speaker Verification (ASV) benchmark task** (cosine similarity
over x-vector-style embeddings, EER as the metric — methodologically
identical to this project's own pipeline), it has a genuinely permissive
**CC0** license, and 1,218 contributors across 12 Indian languages
including Hindi.

**IndicVoices** (and its TTS-oriented derivative IndicVoices-R) is larger
in raw scale (51K speakers / 23.7K hours for IndicVoices; 10,496 speakers /
1,704 hours for IndicVoices-R) and openly licensed for research *and*
commercial use, but it was built for ASR/TTS, not verification — persistent
per-speaker IDs appear to exist (filenames encode speaker/gender for
IndicVoices-R) but recordings-per-speaker and explicit SV suitability are
**UNKNOWN / NOT DOCUMENTED** from what I could verify; this needs direct
inspection of the metadata before use.

**NISP** (IISc/NITK) is small (345 speakers) but is the most clearly
speaker-verification-friendly of the smaller Hindi-relevant datasets: each
speaker recorded ~4–5 minutes *per language* including Hindi and Indian
English, with rich, explicit per-speaker metadata, freely available on
GitHub.

**I-MSV / IITG-MV** (IIT Guwahati) is a purpose-built, multi-sensor Indic
speaker-verification database (the closest thing to an "Indian VoxCeleb"),
but access requires a demand draft/paid request to IIT Guwahati with a
no-redistribution research-only agreement, and its speaker count (100 for
the I-MSV 2022 challenge subset) is modest.

**Recommendation for Hindi: Kathbath as primary (explicit SV task, CC0,
already benchmarked), NISP as a small supplementary/sanity-check set.**

## SECTION 3 — Best Hinglish datasets

This is the weakest category by a significant margin, and I want to be
direct about that rather than padding the list with unsuitable datasets.

Of the candidates investigated:

- **MUCS Hindi-English code-switching data (OpenSLR/104)** has real scale
  (89.86 h train + 5.18 h test) but the challenge's own documentation
  states speaker information was **not available** — only the source
  tutorial video is known, and speaker identity is *assumed* (one speaker
  per tutorial), not confirmed. **Not suitable for constructing verified
  genuine/impostor trials.**
- **HiACC** (2025, peer-reviewed, Zenodo) is the most credible
  purpose-built Hinglish corpus found, but it is small (5.24 hours total)
  and I could not verify its exact speaker count or recordings-per-speaker
  from the available search results — **UNKNOWN**, needs direct dataset
  inspection.
- **IITG-HingCoS** (IIT Guwahati) has 25 hours / ~9,251 code-switched
  sentences, but speaker count and access procedure are **UNKNOWN / NOT
  DOCUMENTED** from what I could verify (likely a restricted-access
  request to IIT Guwahati, similar to their I-MSV/IITG-MV databases, but
  this is inferred, not confirmed).
- **IIITH Hindi-English Code-Mixed Speech Database** is the only Hinglish
  candidate with an explicitly documented speaker count: **78 speakers**,
  but each recorded only **~1 minute** of phonetically-balanced read
  speech, apparently in a single session — this is a real, specific risk
  for genuine-trial construction (see Section 7): one short session per
  speaker may not give two sufficiently independent recordings.
- A "**Hinglish Concatenated Audio Dataset**" was referenced in the task
  but I could not locate it via any authoritative source (official dataset
  page, paper, or dataset-card) — I am not treating it as a verified
  candidate. If you have a specific link/DOI for it, please share it and
  I will audit it directly.

**Recommendation for Hinglish: no candidate here is currently "ready" the
way VoxCeleb or Kathbath are. IIITH Hindi-English Code-Mixed is the most
concrete lead (documented speaker count) but needs direct verification of
whether ≥2 independent recordings exist per speaker; HiACC is the most
credible in terms of publication quality but needs the same direct check.
Treat Hinglish as requiring either (a) direct correspondence with one of
these dataset authors to confirm structure, or (b) building on this
project's own small out-of-domain personal-recording approach
(`scripts/score_personal_recording.py`) as an interim validation set rather
than a full calibration corpus.**

## SECTION 4 — Dataset comparison

### English

| Field | VoxCeleb1 | VoxCeleb2 | LibriSpeech | SITW |
|---|---|---|---|---|
| Purpose | Speaker verification | Speaker verification | ASR (speaker verification is a repurposed/community use) | Speaker verification (real-world/"wild" conditions) |
| Language | English (multi-accent, YouTube) | English (multi-accent, YouTube) | English (read audiobooks) | English (multi-accent, real-world) |
| Approx. hours | 352 h | 2,442 h | ~1,000 h | UNKNOWN (short-to-long segments, 6–180 s each) |
| Approx. utterances | 153,516 | 1,128,246 | UNKNOWN (thousands per speaker split) | 6,445 utterances |
| Number of speakers | 1,251 | 6,112 | 2,000+ | 299 (117 male, 63 female + more; totals vary by release) |
| Recordings per speaker | Many (multiple YouTube videos/segments) | Many | Many (varies widely) | ~8 sessions/person on average |
| Speaker IDs | Yes, persistent, celebrity identity | Yes, persistent | Yes, persistent | Yes, persistent |
| Official trial lists | Yes — VoxCeleb1, VoxCeleb1-E (579,818 trials), VoxCeleb1-H (550,894 trials), cleaned versions | Uses VoxCeleb1 test protocols | No official SV protocol; community-constructed trial lists exist (e.g. ~1.7M-trial lists built by researchers) | Yes — core/assist/multi enrollment-test protocol |
| SV suitability | Very high (this is THE standard SV benchmark) | Very high | Medium (works, but not native) | High, but small scale |
| Audio format/rate | 16 kHz | 16 kHz | 16 kHz | 16 kHz |
| License | Audio access requires accepting VGG's data-access terms; commonly described as "research/non-commercial use," with some VoxCeleb2 metadata under CC BY-SA 4.0 — **treat overall audio license as LICENSE PARTIALLY UNCLEAR, verify the exact current terms on the official request form before use** | Same as VoxCeleb1 | CC BY 4.0 (fully permissive) | Requires request/agreement via SRI International |
| Access method | Request form on official site | Request form on official site | Direct download (OpenSLR) | Request via SRI |
| Download size | ~30–40 GB range typically cited for compressed audio (**not independently verified here — treat as approximate**) | Substantially larger than VoxCeleb1 (**exact GB not verified here**) | ~60 GB (all subsets) | Smaller (hours are short) — **exact GB UNKNOWN** |
| Strengths | Official trial protocol, huge speaker count, already integrated into this project's Phase 7 pipeline | Even larger speaker/utterance count | Trivial to obtain, extremely permissive license | Genuinely adversarial "wild" conditions for robustness testing |
| Weaknesses | Access-gated, YouTube-content provenance, model was trained on it (see docs/model.md Phase 7 caveat) | Same as VoxCeleb1, larger download | No official SV protocol; clean/read speech is not very "wild" | Small, license friction, not enough scale to be a primary calibration source |
| Recommended use | **Primary English calibration/evaluation source** | Optional scale-up once VoxCeleb1 pipeline is proven | Optional easy-license supplement / cross-check | Optional small hard-evaluation supplement |

### Hindi / Indian

| Field | Kathbath (AI4Bharat/IndicSUPERB) | IndicVoices / IndicVoices-R (AI4Bharat) | NISP (IISc/NITK) | I-MSV / IITG-MV (IIT Guwahati) |
|---|---|---|---|---|
| Purpose | Multi-task Indian speech benchmark, **includes an explicit ASV task** | ASR (IndicVoices) / TTS (IndicVoices-R) | Speaker profiling (accent/native-language + speaker traits) | Speaker verification (purpose-built) |
| Language | 12 Indian languages incl. Hindi | 22 Indian languages incl. Hindi | Hindi, Kannada, Malayalam, Tamil, Telugu, + Indian-accented English | 13 Indian languages (I-MSV 2022 subset) |
| Approx. hours | 1,684 h total across 12 languages (**Hindi-only hours not confirmed from sources reviewed — UNKNOWN**) | 23.7K h (IndicVoices) / 1,704 h (IndicVoices-R), 9–175 h per language | UNKNOWN exact hours; ~4–5 min/speaker/language | ~100 h (I-MSV 2022 subset) |
| Approx. utterances | UNKNOWN exact per-language count | UNKNOWN precise count (IndicVoices); large (IndicVoices-R, derived from ASR transcripts) | 28,268 total utterances | UNKNOWN exact |
| Number of speakers | 1,218 total contributors across 12 languages (**Hindi-only speaker count NOT confirmed — UNKNOWN**) | 51,000 (IndicVoices) / 10,496 (IndicVoices-R) | 345 (219 male, 126 female) | 100 (I-MSV 2022 challenge subset) |
| Recordings per speaker | UNKNOWN exact; benchmark implies multiple utterances per speaker (needed for the ASV task to work at all) | UNKNOWN precise; large hours/speaker ratio for IndicVoices suggests few minutes per speaker on average | Multiple utterances per language per speaker (part of the 28,268 total) | 2 sessions per speaker (reading + conversation mode) |
| Speaker IDs | Yes, persistent (used directly for the ASV benchmark) | Yes, persistent (speaker/gender encoded in filenames for IndicVoices-R; IndicVoices fields not fully confirmed — UNKNOWN) | Yes, persistent, with rich demographic metadata | Yes, persistent |
| Official trial lists | Yes — IndicSUPERB publishes ASV metadata files (`valid_data.txt`, `test_known_data.txt`, `test_data.txt`); **exact genuine/impostor pairing format not independently confirmed from sources reviewed — verify directly from the repo before use** | No official SV trial list found | No official SV trial list found | Provided as part of the I-MSV/IITG-MV challenge packages |
| SV suitability | High — the only Hindi-relevant dataset with a documented, benchmarked ASV task | Medium — plausible but unverified for SV specifically | Medium — designed for profiling, not verification, but structurally usable | High — purpose-built for SV |
| Audio format/rate | UNKNOWN exact (typical crowdsourced 16 kHz assumed, not confirmed) | UNKNOWN exact per dataset | UNKNOWN exact | UNKNOWN exact |
| License | **CC0** ("no rights reserved") | Described as "open license for research and commercial use" (IndicVoices) — exact terms should be re-confirmed on the dataset card before use | Publicly hosted on GitHub; explicit license terms **NOT CONFIRMED — LICENSE UNCLEAR**, check repo before use | Requires demand draft + signed research-only, no-redistribution agreement to IIT Guwahati |
| Access method | Hugging Face (`ai4bharat/Kathbath`) | Hugging Face / GitHub / AI4Bharat site | GitHub (`iiscleap/NISP-Dataset`) | Paid request form + physical/administrative process |
| Download size | UNKNOWN exact (1,684 h implies tens of GB) | Very large (23.7K h implies hundreds of GB for IndicVoices; smaller for IndicVoices-R) | Small (28,268 short utterances) | UNKNOWN exact |
| Strengths | Explicit SV benchmark task, CC0, already includes Hindi | Massive scale, open license, multiple Indian languages | Rich per-speaker metadata, easy access, includes Hindi + Indian English | Purpose-built for SV, multi-sensor realism |
| Weaknesses | Hindi-specific breakdown not confirmed from sources reviewed; needs direct dataset inspection | Not verification-native; per-speaker recording count for genuine trials unconfirmed | Small scale, license unclear | Costly/slow access process, small speaker count, no free digital download confirmed |
| Recommended use | **Primary Hindi calibration/evaluation source** (pending direct confirmation of Hindi-only speaker/utterance counts) | Optional scale-up source once license/fields are directly confirmed | Small supplementary/sanity-check set | Optional purpose-built supplement if the access process is acceptable |

### Hinglish

| Field | HiACC | MUCS Hindi-English (OpenSLR/104) | IITG-HingCoS | IIITH Hindi-English Code-Mixed |
|---|---|---|---|---|
| Purpose | Code-switched Hinglish speech (adults + children), first of its kind | Code-switching ASR challenge data | Code-switching ASR database | Phonetically-balanced code-mixed ASR database |
| Language | Hindi-English code-switched | Hindi-English code-switched | Hindi-English code-switched | Hindi-English code-switched |
| Approx. hours | 5.24 h | 89.86 h train + 5.18 h test | 25 h | UNKNOWN exact (78 speakers × ~1 min ≈ ~1.3 h) |
| Approx. utterances | 3,318 adult + 1,858 child segments | UNKNOWN exact utterance count | 9,251 code-switching sentences | UNKNOWN exact |
| Number of speakers | UNKNOWN / NOT DOCUMENTED from sources reviewed | Speaker info explicitly **not available**; only source video/tutorial is known | UNKNOWN / NOT DOCUMENTED from sources reviewed | **78** (explicitly documented) |
| Recordings per speaker | UNKNOWN | UNKNOWN (assumed ~1 "speaker" per tutorial, unconfirmed) | UNKNOWN | ~1 minute, apparently single session — **insufficient documented evidence of ≥2 independent recordings per speaker** |
| Speaker IDs | UNKNOWN whether persistent/anonymized IDs are published | **Not available** — explicitly documented limitation | UNKNOWN | Yes, speaker-indexed, but session independence UNKNOWN |
| Official trial lists | No | No | No | No |
| SV suitability | **UNKNOWN — needs direct dataset inspection** | **Low / not suitable** (no reliable speaker labels) | **UNKNOWN — needs direct dataset inspection** | **Low-to-medium** — has speaker IDs but single-session recordings raise genuine-trial independence concerns |
| Audio format/rate | UNKNOWN exact | 16 kHz, 16-bit | UNKNOWN exact | UNKNOWN exact |
| License | Zenodo-hosted, open for research (exact license terms should be re-confirmed on the record page) | Distributed under a Microsoft Open Data license (per OpenSLR listing) — **verify exact redistribution/commercial-use terms before relying on it** | UNKNOWN / NOT DOCUMENTED — LICENSE UNCLEAR | UNKNOWN / NOT DOCUMENTED — LICENSE UNCLEAR |
| Access method | Zenodo direct download | OpenSLR direct download | UNKNOWN — likely a request to IIT Guwahati (inferred from sibling databases, not confirmed) | UNKNOWN — likely a request to IIIT Hyderabad's speech lab (inferred, not confirmed) |
| Download size | Small (5.24 h) | Moderate (~95 h combined) | Moderate (25 h) | Very small (~1.3 h) |
| Strengths | Most credible peer-reviewed Hinglish-specific corpus found; publicly downloadable | Real scale, free direct download | Real scale, dedicated Hinglish focus | Only candidate with an explicitly documented speaker count |
| Weaknesses | Speaker count/structure unconfirmed | No usable speaker labels — ruled out for SV | Speaker count/access unconfirmed | Extremely short (~1 min) apparently-single-session recordings per speaker |
| Recommended use | Candidate pending direct verification of speaker structure | **Not recommended for speaker verification** (usable only for the code-switching ASR purpose it was built for) | Candidate pending direct verification | Candidate pending direct verification of multi-session recordings; otherwise use only as a small out-of-domain sanity check, not a calibration source |

## SECTION 5 — Licensing/access comparison

| Dataset | Registration required? | Research use allowed? | SV research allowed? | Redistribution | Embeddings storable/shareable? | Status |
|---|---|---|---|---|---|---|
| VoxCeleb1/2 | Yes (request form) | Yes | Yes (this is its primary use case) | Not permitted (raw audio) | Derived embeddings are commonly shared in the SV research community, but the *official* terms should be re-read before any redistribution — **verify current terms at request time** | Access-gated, terms partially unclear on redistribution of derivatives |
| LibriSpeech | No | Yes | Yes (community-repurposed) | Yes — CC BY 4.0 is very permissive | Yes | Clear |
| SITW | Yes (request to SRI) | Yes | Yes (this is its purpose) | UNKNOWN exact terms | UNKNOWN | LICENSE UNCLEAR (exact terms not reviewed here) |
| Kathbath | No (direct HF download) | Yes | Yes (documented SV benchmark) | Yes — **CC0** | Yes | Clear (most permissive dataset in this audit) |
| IndicVoices / IndicVoices-R | No (direct HF/GitHub download) | Yes | Plausible, not explicitly confirmed for SV | Described as open for research **and commercial** use — re-confirm on the dataset card | Plausible | Mostly clear, re-verify before commercial use |
| NISP | No (GitHub) | Presumed yes (research dataset) | Not explicitly confirmed | UNKNOWN | UNKNOWN | LICENSE UNCLEAR |
| I-MSV / IITG-MV | Yes (paid demand draft + form) | Yes, research-only | Yes (purpose-built for SV) | Explicitly **not permitted** | UNKNOWN | Clear but restrictive (no redistribution, paid access) |
| HiACC | No (Zenodo) | Yes | UNKNOWN | UNKNOWN exact Zenodo license field | UNKNOWN | LICENSE UNCLEAR (re-check Zenodo record) |
| MUCS Hindi-English | No (OpenSLR) | Yes | Not suitable for SV regardless of license (no speaker labels) | Microsoft Open Data license terms — re-verify | N/A given SV unsuitability | Not a primary blocker since dataset is unsuitable for SV anyway |
| IITG-HingCoS | UNKNOWN | Presumed yes | UNKNOWN | UNKNOWN | UNKNOWN | LICENSE UNCLEAR |
| IIITH Hindi-English Code-Mixed | UNKNOWN | Presumed yes (academic dataset) | UNKNOWN | UNKNOWN | UNKNOWN | LICENSE UNCLEAR |

## SECTION 6 — Speaker-ID reliability

Using the Section 4 (Phase 7B spec) checklist per dataset:

| Dataset | Speaker IDs provided? | Persistent across recordings? | Multiple recordings/speaker? | Recordings independent enough? | Genuine trials constructible? | Impostor trials constructible? | Documentation permits SV use? |
|---|---|---|---|---|---|---|---|
| VoxCeleb1/2 | Yes | Yes | Yes (many) | Yes (different videos/sessions) | Yes | Yes | Yes (explicit purpose) |
| LibriSpeech | Yes | Yes | Yes | Yes (different chapters/books) | Yes (community-established) | Yes | Implicitly yes (widely used this way) |
| SITW | Yes | Yes | Yes (~8 sessions avg) | Yes | Yes | Yes | Yes (explicit purpose) |
| Kathbath | Yes | Yes | UNKNOWN exact count, but implied by the ASV benchmark's existence | Presumed yes (benchmark wouldn't work otherwise) | Yes (used for this exact purpose) | Yes | Yes (explicit ASV benchmark) |
| IndicVoices/-R | Yes | Yes | UNKNOWN exact count | UNKNOWN | Plausible, unconfirmed | Yes (many speakers) | Not explicitly confirmed for SV |
| NISP | Yes | Yes | Yes (per language) | Likely yes (same recording session per language, though — see caveat below) | Plausible | Yes | Not stated explicitly, but no restriction found |
| I-MSV/IITG-MV | Yes | Yes | Yes (2 sessions: reading + conversation) | Yes | Yes | Yes | Yes (explicit purpose) |
| HiACC | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| MUCS Hindi-English | **No** (explicitly not available) | N/A | N/A | N/A | **No** | **No** | No — dataset documentation explicitly says speaker info isn't available |
| IITG-HingCoS | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| IIITH Hindi-English Code-Mixed | Yes | UNKNOWN (appears single-session) | UNKNOWN — likely just one ~1 min recording per speaker | UNKNOWN, likely **no** (single session) | UNKNOWN, likely **not without more sessions** | Yes (different speakers still available) | Not stated explicitly |

**NISP caveat**: NISP's "multiple recordings per speaker" are multiple
*languages* per speaker (Hindi, English, etc.) in what appears to be
contiguous recording sessions, not necessarily independent sessions
recorded on different days — genuine trials from NISP should ideally pair
two Hindi (or two English) recordings of the same speaker if the corpus
structure supports it; this needs to be confirmed by inspecting the actual
file layout before building trials, not assumed from documentation alone.

## SECTION 7 — Genuine/impostor trial feasibility

Following the Phase 7A `Trial` model already implemented (`reference_id`,
`test_id`, `label`, `reference_speaker`, `test_speaker`) and the explicit
instruction in this phase (Section 5 of the request) that cross-dataset
speaker identity must never be assumed:

- **Within VoxCeleb (English)**: fully feasible today with the existing
  Phase 7 pipeline, unchanged.
- **Within Kathbath (Hindi)**: feasible in principle (persistent speaker
  IDs, explicit ASV precedent) but requires writing a new
  `src/speaker_verification/datasets/kathbath.py` adapter (analogous to
  `voxceleb.py`) once the exact file/metadata layout is confirmed — not
  done in this phase, per the instruction not to modify the codebase
  unnecessarily.
- **Within NISP (Hindi/Indian-English)**: feasible with caveats above;
  small enough to be a supplementary/sanity-check set rather than a
  primary source.
- **Within Hinglish candidates**: not currently feasible with confidence
  for any candidate audited — either speaker IDs are absent (MUCS) or
  unconfirmed (HiACC, IITG-HingCoS) or the per-speaker session
  independence is doubtful (IIITH Code-Mixed). This is a genuine gap, not
  a solvable-by-code problem.
- **Cross-language genuine trials** (e.g., English↔Hindi, Hindi↔Hinglish)
  are **not constructible** with any combination audited here, because
  none of these datasets establish that "speaker X in VoxCeleb" is the
  same real person as "speaker Y in Kathbath" (or any other pair). Per
  Section 5 of the request, these must be treated as impostor-only or
  simply not constructed across datasets at all — only within-dataset
  genuine trials are valid unless a dataset explicitly links identities
  across corpora (none found here do).
- **Cross-language impostor trials** (pairing a VoxCeleb English speaker
  against a Kathbath Hindi speaker, for example) ARE constructible and
  valid — impostor trials only require that the two speakers be
  different real people, which is a safe assumption across independently
  collected datasets. These could be a useful additional impostor pool to
  test whether language mismatch itself affects the false-accept rate,
  as a separate, clearly-labeled analysis — not mixed into the primary
  within-language calibration/evaluation pools.

## SECTION 8 — Recommended combination

**Recommended: 3 datasets — VoxCeleb1 (English) + Kathbath (Hindi) + a
still-to-be-confirmed Hinglish source, evaluated separately per language
rather than pooled into one undifferentiated set.**

Reasoning:

- VoxCeleb1 and Kathbath both clear the bar this audit set (speaker-ID
  reliability, multiple recordings per speaker, explicit or de facto SV
  suitability, workable licensing) — English and Hindi each get one solid
  primary source rather than a compromise.
- No Hinglish dataset audited here clears that same bar today. Recommending
  one anyway just to fill the third slot would violate the explicit
  instruction not to assume a "Hinglish dataset" is automatically an
  "SV dataset." The honest recommendation is: **pick a Hinglish candidate
  only after direct inspection of its actual file/metadata structure**
  (HiACC and IIITH Code-Mixed are the two worth inspecting first), and in
  the meantime treat this project's own personal-recording pipeline
  (`scripts/score_personal_recording.py`) as the interim Hinglish/domain
  sanity check.
- NISP is recommended as a small supplementary Hindi/Indian-English set
  (not a replacement for Kathbath) specifically because of its rich,
  explicit per-speaker metadata, which is useful for later
  demographic-slice analysis (e.g., does EER vary by gender or region?) —
  optional, not required for a first pass.

This is 2 firm primary datasets + 1 deferred (Hinglish) + 1 optional
supplement (NISP), which fits the request's explicit allowance of "2, 3,
or 4 datasets if that produces a better scientifically defensible
benchmark" — here, a defensible 2-language benchmark now is better than a
padded 3-language benchmark built on an unverified Hinglish source.

## SECTION 9 — Recommended dataset size

Derived from what's actually documented (not invented), and deliberately
conservative relative to the ~1,000-speaker target ranges given in the
request, because Phase 7H already established the project's own precedent
of starting small and scaling up only after the pipeline is proven:

| Language | Speakers (initial target) | Trials (initial target) | Basis |
|---|---|---|---|
| English (VoxCeleb1) | 300–500 speakers | ~1,000 genuine + ~1,000 impostor | Matches this project's existing Phase 7H guidance; VoxCeleb1's 1,251 speakers comfortably support this without needing VoxCeleb2 |
| Hindi (Kathbath) | As many as the Hindi-specific subset actually contains (**must be confirmed by direct inspection — total across 12 languages is 1,218 contributors, Hindi-only count UNKNOWN**), capped at a similar ~300–500-speaker order of magnitude if the full Hindi subset is larger | ~1,000 genuine + ~1,000 impostor, or fewer if the Hindi subset is smaller than that | Same scale rationale as English, contingent on real numbers once confirmed |
| Hindi (NISP, optional supplement) | Up to all 345 speakers | Limited by NISP's small per-speaker utterance count — likely a few hundred trials at most | Small dataset; useful as a cross-check, not a primary source |
| Hinglish | Not sized yet — pending dataset selection | Not sized yet | Deferred per Section 8 |

I have deliberately not invented a Hinglish number, per the explicit
instruction in Section 13 of the request ("Do not invent these numbers").

## SECTION 10 — Recommended calibration/evaluation split

Use the **same speaker-disjoint split methodology already implemented and
tested** in `src/speaker_verification/datasets/trials.py`
(`split_trials_by_speaker`), unchanged:

- Split unique speakers (not trials) into calibration/evaluation sides via
  a seeded shuffle.
- Keep a trial only if both its speakers are on the same side; drop
  cross-side trials (already the documented, tested behavior).
- Re-verify zero speaker overlap with `speaker_overlap()` before trusting
  any resulting FAR/FRR numbers (already the documented, tested behavior).

**Per-language split, not one pooled split**: because cross-language
genuine trials aren't constructible (Section 7), calibration and
evaluation should each be computed **per language** (English calibration
→ English evaluation; Hindi calibration → Hindi evaluation), producing
separate EER/threshold numbers per language rather than one pooled number
that would obscure language-specific behavior. This directly feeds
Section 16 of the request (global vs. per-language calibration) — see
below.

## SECTION 11 — Expected disk/storage requirements

Storage estimates for the *proposed initial subset sizes* (Section 9), not
the full datasets:

- **Audio for ~500 English speakers' worth of VoxCeleb1 utterances
  (subset, not the full 352 h)**: VoxCeleb1 utterances are typically a
  few seconds each; a ~2,000-trial subset (drawing on a few thousand
  unique utterances) is very unlikely to exceed a few GB — **exact figure
  UNKNOWN without downloading**, but an order of magnitude below the full
  ~30–40 GB VoxCeleb1 archive.
- **Audio for a comparable Hindi (Kathbath) subset**: similarly small,
  likely a few GB, once the actual Hindi-only subset is identified and
  filtered (Kathbath's full 1,684 h across 12 languages implies each
  language is on the order of ~100+ hours, so a small speaker/utterance
  subset is a small fraction of that).
- **Embedding cache**: per Phase 7's existing design
  (`embedding_cache.py`), 192-dimensional float32 embeddings are ~768
  bytes each; even 10,000 cached embeddings (well beyond this phase's
  target scale) is only ~7.7 MB — negligible compared to audio.
- **Overall**: a combined English+Hindi initial subset at the sizes in
  Section 9 should comfortably fit in low-single-digit GB, well under the
  "~6 GB total" ballpark mentioned in the request — but this is a
  reasoned estimate, not a measured one, since no data was downloaded in
  this phase.

## SECTION 12 — Why this combination is better than using one 40-language dataset

A single, giant multi-language dataset (e.g., trying to force IndicVoices'
22-language, 51K-speaker scale, or an even broader multilingual corpus, to
serve as the whole calibration source) would work against every priority
this audit was asked to optimize for:

1. **Speaker-verification suitability is diluted.** A dataset built for
   ASR/TTS coverage across dozens of languages optimizes for linguistic
   breadth, not for "≥2 independent, verifiable recordings per speaker" —
   exactly the property this audit found IndicVoices' documentation does
   *not* clearly establish, while Kathbath's narrower, ASV-benchmarked
   scope does.
2. **It reintroduces the cross-dataset-identity problem, not the
   cross-language one.** A 40-language dataset would still be *one*
   dataset with *one* internally-consistent speaker-ID scheme — so it
   doesn't actually solve the real difficulty here, which is that no
   dataset in this audit spans English + Hindi + Hinglish with linked
   speaker identities. The choice is never "one big dataset avoids the
   cross-dataset problem" — no candidate audited avoids it.
3. **Per-language calibration is unaffected by pooling more unrelated
   languages in.** Since Section 10 recommends per-language calibration
   splits, adding 37 languages the project doesn't target contributes
   nothing to English/Hindi/Hinglish threshold quality — it only adds
   download size, filtering complexity, and license surface area to
   review.
4. **Licensing risk compounds with scale for no benefit.** Each additional
   dataset/language is another license to verify (Section 5); a giant
   multi-language corpus is not one license decision, it's dozens,
   without improving the three languages actually in scope.
5. **This project's own precedent (Phase 7H) is small-first.** The
   generic VoxCeleb pipeline was deliberately built and tested at
   ~1,000-trial scale before considering VoxCeleb2's full 6,112 speakers.
   Reaching for a 40-language dataset here would abandon that discipline
   for a dataset whose extra breadth doesn't serve the actual B.S.
   Detector target domain (English, Hindi, Hinglish — not Bengali, Tamil,
   Telugu, etc., however useful those might be to a different project).

## SECTION 13 — Risks/limitations

- **Several key statistics are UNKNOWN**, most importantly Kathbath's
  Hindi-only speaker/utterance/hour breakdown and every Hinglish
  candidate's exact speaker count — these must be confirmed by directly
  inspecting the dataset (metadata files, not just papers/search results)
  before finalizing a subset size.
- **No dataset in this audit was actually downloaded or opened** — this
  audit is based on official pages, papers, and dataset cards. It is
  possible (though I have tried to be careful) that a metadata field or
  license term has changed since the cited source was published, or that
  a search result summarized a source imperfectly. Direct inspection
  before committing to a dataset is still required.
- **VoxCeleb's license terms were reported inconsistently** across
  sources found (permissive CC-style language for metadata vs.
  request-gated/non-commercial framing for audio access) — this needs to
  be resolved by reading the actual current request-form terms at
  download time, not assumed from this audit.
- **Model-domain overlap**: the ECAPA-TDNN checkpoint this project uses
  was itself trained on VoxCeleb (already documented in `docs/model.md`
  from Phase 7) — English calibration numbers from VoxCeleb are
  best-case/in-domain for the model, not an independent test.
- **No dataset audited here explicitly links speaker identity across
  languages** — this is expected (per Section 5 of the request) but means
  a true "does this person's Hindi voice match their English voice"
  evaluation is not currently possible with any combination found. That
  would require either a bespoke recording effort or a not-yet-found
  dataset explicitly designed for cross-lingual speaker verification.
- **Hinglish remains the weakest link** by a wide margin — every
  candidate has an open question (speaker IDs missing, speaker count
  undocumented, or single-session recordings), and this audit did not
  find a dataset that clears the same bar VoxCeleb/Kathbath do.
- **Access friction**: I-MSV/IITG-MV requires a paid, physical
  demand-draft process; VoxCeleb requires a request-form approval; these
  add lead time beyond a simple `wget`/`git clone`, which should be
  planned for.

## SECTION 14 — Exact next step

This phase stops here, per your instructions. If/when you want to proceed,
the concrete next actions (still not started) would be:

1. **You obtain access** to VoxCeleb1 (already documented in
   `docs/calibration.md` from Phase 7) and confirm the exact current
   VoxCeleb license/redistribution terms on the request form.
2. **Directly inspect Kathbath's metadata** (via the Hugging Face dataset
   viewer or a small metadata-only download) to confirm: Hindi-only
   speaker count, utterances-per-speaker, and the exact file-naming/ID
   scheme, so a `src/speaker_verification/datasets/kathbath.py` adapter
   can be designed analogously to the existing `voxceleb.py`.
3. **Directly inspect HiACC and IIITH Hindi-English Code-Mixed** metadata
   (both are freely downloadable/inspectable without a paid request) to
   determine whether either has ≥2 independent recordings per speaker,
   resolving Section 3/7's open Hinglish question before committing to
   either.
4. Only after 1–3 are confirmed with real numbers: design and implement
   the Kathbath (and, if viable, Hinglish) dataset adapter(s), following
   the exact pattern Phase 7A already established — a dataset-specific
   parser producing the existing dataset-agnostic `Trial` type, with no
   changes to `calibration.py`, `verifier.py`, `similarity.py`, or
   `encoder.py`.

**No datasets have been downloaded. No embeddings have been generated. No
EER or threshold has been calculated. Phase 8 has not been started. This
document (`docs/dataset_audit_phase7b.md`) has been written to disk but
not committed to Git, pending your review.**

---

### Sources consulted

- [VoxCeleb1 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
- [VoxCeleb2 dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
- [The Speakers in the Wild (SITW) Speaker Recognition Database — SRI](https://www.sri.com/publication/speech-natural-language-pubs/the-speakers-in-the-wild-sitw-speaker-recognition-database/)
- [SRI International — SITW Speaker Recognition Challenge](http://www.speech.sri.com/projects/sitw/)
- [openslr/librispeech_asr · Datasets at Hugging Face](https://huggingface.co/datasets/openslr/librispeech_asr)
- [I-MSV 2022: Indic-Multilingual and Multi-sensor Speaker Verification Challenge (arXiv)](https://arxiv.org/pdf/2302.13209)
- [MSV Challenge 2022: NPU-HC Speaker Verification System for Low-resource Indian Languages](https://arxiv.org/abs/2211.16694)
- [IITG-MV SR Database — EMST Lab, IIT Guwahati](https://www.iitg.ac.in/eee/emstlab/SRdatabase/introduction.php)
- [ai4bharat/indicvoices_r · Datasets at Hugging Face](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
- [IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus (arXiv)](https://arxiv.org/pdf/2409.05356)
- [IndicVoices — AI4Bharat](https://ai4bharat.iitm.ac.in/datasets/indicvoices/)
- [ai4bharat/IndicVoices · Datasets at Hugging Face](https://huggingface.co/datasets/ai4bharat/IndicVoices)
- [README.md · ai4bharat/Kathbath at main](https://huggingface.co/datasets/ai4bharat/Kathbath/blob/main/README.md)
- [GitHub - AI4Bharat/IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB)
- [IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian Languages (arXiv)](https://arxiv.org/pdf/2208.11761)
- [NISP: A Multi-lingual Multi-accent Dataset for Speaker Profiling (arXiv)](https://arxiv.org/pdf/2007.06021)
- [GitHub - iiscleap/NISP-Dataset](https://github.com/iiscleap/NISP-Dataset)
- [MUCS 2021: Multilingual and Code-Switching ASR Challenges](https://navana-tech.github.io/MUCS2021/challenge_details.html)
- [Multilingual and code-switching ASR Challenge Dataset — OpenSLR](https://www.openslr.org/104/)
- [Multilingual and code-switching ASR challenges for low resource Indian languages (arXiv)](https://arxiv.org/pdf/2104.00235)
- [HiACC: Hinglish adult & children code-switched corpus — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925006109)
- [HiACC: Hinglish adult & children code-switched corpus — PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12329218/)
- [IITG-HingCoS Corpus: A Hinglish Code-Switching Database — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167639318304217)
- [Development of IIITH Hindi-English Code Mixed Speech Database (ISCA Archive)](https://www.isca-archive.org/sltu_2018/rambabu18_sltu.pdf)
- [IIIT-H Indic Speech Databases](http://speech.iiit.ac.in/index.php/research-svl/69.html)
- [Fisher English Training Speech Part 1 — LDC Catalog](https://catalog.ldc.upenn.edu/LDC2004S13)
- [Fisher English Training Part 2, Speech — LDC Catalog](https://catalog.ldc.upenn.edu/LDC2005S13)
- [mozilla-foundation/common_voice_11_0 · Datasets at Hugging Face](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
- [OpenSLR/103 — Multilingual and code-switching ASR Challenge Dataset](https://www.openslr.org/103/)
- [Vaani: Capturing the language landscape for an inclusive digital India (arXiv)](https://arxiv.org/html/2603.28714v1)
