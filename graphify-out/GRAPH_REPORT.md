# Graph Report - B.S-Detector  (2026-10-30)

## Corpus Check
- 158 files · ~138,170 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 38 file(s) not represented in the graph (top: (none) 15, .css 14, .pth 4)

## Summary
- 2010 nodes · 4397 edges · 112 communities (94 shown, 18 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 437 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `af8007df`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pytest
- experiment.py
- EmbeddingCache
- pathlib
- experiment.py
- audio.py
- run.py
- encoder.py
- test_datasets_voxceleb.py
- Steep — Style Reference
- test_multi_reference.py
- audio.ts
- sv_duration.py
- SpeakerEncoder
- sv_benchmark.py
- run_calibration
- Trial
- package.json
- Findings.tsx
- check_head.py
- run_indicsynth_finetune.py
- Analyzer.tsx
- Speaker Verification
- calibration.py
- Rules
- App.tsx
- train_sls_channel_robust.py
- test_datasets_trials.py
- backend.py
- train_sls_robust.py
- compilerOptions
- VoiceInput.tsx
- train_phase12.py
- app.py
- compute_far
- speaker-a-1
- sv_models.py
- encoder.py
- run_forensic_detection
- compilerOptions
- Audio Deepfake Detection: SLS on Frozen XLS-R 300M (Hindi Audio Benchmark)
- verify
- verification
- sys
- test_sliding_window.py
- inference.py
- B.S-Detector: Parameter-Efficient Audio Deepfake Detection on Hindi Speech
- InsufficientCalibrationDataError
- detection
- referenceDetection
- Hero.tsx
- B.S. Detector
- train_ecapa_indian.py
- generate_figures.py
- peakSpoofWindow
- eval_ecapa_comparison.py
- EncoderError
- fake-hindi-xtts
- real-hindi
- speaker-a-2
- speaker-b-1
- json
- deepfake_check.py
- numpy
- different-speaker
- Phase 7B — Dataset Discovery, Audit & Selection (English / Hindi / Hinglish)
- score_recordings.py
- build_trials.py
- Agent Handoff — B.S-Detector (Audio Deepfake Detection)
- test_phone_band.py
- ResultCard.tsx
- Recording protocol: real-world test set
- server
- SLSHead
- calibrate
- Phase 9: Speaker verification on Hindi and Indian-English speakers
- .oxlintrc.json
- load_hindi_head
- ui_path_check.py
- Demo recordings
- render_review.py
- Stage 1 (deepfake detection): measured gaps and improvement plan
- Sample audio credits
- check_environment.py
- Phase 7D — Dataset Access & Final Verification
- demo/manifest.json
- tsconfig.json
- metrics.py
- AGENTS.md
- VerifierState
- load_speaker_name_map
- Calibration and the verification decision (Phase 6)
- scripts/test_encoder.py
- test_app_speaker.py
- features.py
- compute_score_statistics
- Phase 7: large-scale calibration using VoxCeleb
- Phase 8C — Full VoxCeleb1 official-protocol benchmark
- Ecapa
- load_cached_audio
- Large-scale VoxCeleb calibration (Phase 7)
- Results
- Setup

## God Nodes (most connected - your core abstractions)
1. `SpeakerEncoder` - 45 edges
2. `cosine_similarity()` - 41 edges
3. `calibrate()` - 38 edges
4. `EmbeddingCache` - 37 edges
5. `SpeakerVerifier` - 30 edges
6. `parse_trial_list()` - 26 edges
7. `run_calibration()` - 24 edges
8. `_StubEncoder` - 22 edges
9. `preprocess_audio()` - 21 edges
10. `Trial` - 20 edges

## Surprising Connections (you probably didn't know these)
- `1. Data` --references--> `train()`  [INFERRED]
  docs/speaker-verification-hindi-english.md → scripts/train_ecapa_indian.py
- `1. Kathbath` --references--> `Trial`  [INFERRED]
  docs/dataset_access_phase7d.md → src/speaker_verification/datasets/trials.py
- `Deterministic subset sampling (Phase 7B)` --references--> `sample_trials()`  [INFERRED]
  docs/calibration.md → src/speaker_verification/datasets/trials.py
- `4. Side-by-side comparison` --references--> `split_trials_by_speaker()`  [INFERRED]
  docs/dataset_metadata_inspection_phase7c.md → src/speaker_verification/datasets/trials.py
- `Hardware / device support` --references--> `resolve_device()`  [INFERRED]
  docs/speaker-verification.md → src/speaker_verification/encoder.py

## Import Cycles
- None detected.

## Communities (112 total, 18 thin omitted)

### Community 0 - "pytest"
Cohesion: 0.06
Nodes (57): load(), main(), cosine_similarity(), EmbeddingMismatchError, InvalidEmbeddingError, SimilarityError, _validate_embedding(), InvalidThresholdError (+49 more)

### Community 1 - "experiment.py"
Cohesion: 0.13
Nodes (9): extract_and_cache_features(), load_audio(), main(), predict(), extract_hidden_states(), find_speech_onset_index(), load_backbone(), pad_or_truncate() (+1 more)

### Community 2 - "EmbeddingCache"
Cohesion: 0.05
Nodes (37): write_trials_csv(), collect_unique_utterances(), extract_embeddings(), main(), f(), main(), _read_trials_csv(), score_trials() (+29 more)

### Community 3 - "pathlib"
Cohesion: 0.11
Nodes (11): collect_hindi_metadata(), main(), print_diagnostics(), stratified_sample(), generate_metadata_parquet(), main(), create_fewshot_pools(), create_test_split() (+3 more)

### Community 4 - "experiment.py"
Cohesion: 0.16
Nodes (7): evaluate_zeroshot(), run_fewshot_experiment(), _run_inference(), train_fewshot(), collate_features(), load_sls_checkpoint(), SLSHead

### Community 5 - "audio.py"
Cohesion: 0.09
Nodes (32): Audio preprocessing (Phase 3 — implemented), Audio preprocessing, Manually inspecting a single file, Running the audio preprocessing tests, main(), AudioDecodeError, AudioError, AudioFileNotFoundError (+24 more)

### Community 6 - "run.py"
Cohesion: 0.26
Nodes (8): ensure_environment(), ensure_frontend(), ensure_hindi_head(), main(), open_when_up(), port_free(), say(), venv_python()

### Community 8 - "encoder.py"
Cohesion: 0.12
Nodes (10): Embedding output, _explain_missing_trials_file(), main(), _read_trials(), main(), _encode(), main(), main() (+2 more)

### Community 9 - "test_datasets_voxceleb.py"
Cohesion: 0.10
Nodes (29): DatasetError, normalize_utterance_id(), parse_trial_list(), resolve_utterance_path(), speaker_id_from_utterance_id(), TrialListParseError, test_normalize_backslash_paths_are_handled(), test_normalize_converts_speaker_name_form() (+21 more)

### Community 10 - "Steep — Style Reference"
Cohesion: 0.05
Nodes (37): Accent Peach Card, Agent Prompt Guide, Avatar Bubble, Border Radius, Components, CSS Custom Properties, Do, Do's and Don'ts (+29 more)

### Community 11 - "test_multi_reference.py"
Cohesion: 0.09
Nodes (19): test_index_html_is_revalidated(), client(), Ecapa, features(), post(), Second, test_a_bad_extra_clip_is_named(), test_more_than_three_clips_are_refused() (+11 more)

### Community 12 - "audio.ts"
Cohesion: 0.17
Nodes (17): computeSpectrogram(), decodeToClip(), encodeWav(), fftInPlace(), LUT, MAX_SECONDS, paintSpectrogram(), SAMPLE_RATE (+9 more)

### Community 13 - "sv_duration.py"
Cohesion: 0.13
Nodes (11): condition(), embed(), fit(), fit_threshold(), groups(), load(), main(), pair_trials() (+3 more)

### Community 14 - "SpeakerEncoder"
Cohesion: 0.13
Nodes (22): Cosine similarity, Real ECAPA integration test (two audio files), Running the similarity tests, Similarity limitations, InvalidWaveformError, SpeakerEncoder, _make_sine_wave(), speaker_encoder() (+14 more)

### Community 15 - "sv_benchmark.py"
Cohesion: 0.13
Nodes (21): Backend, corpora(), _cv_files(), cv_speakers(), _decode(), eer(), embed(), evaluate() (+13 more)

### Community 16 - "run_calibration"
Cohesion: 0.14
Nodes (22): _format_report(), main(), _median(), _read_scores(), run_calibration(), _speakers(), _utterances(), _row() (+14 more)

### Community 17 - "Trial"
Cohesion: 0.15
Nodes (10): Speaker leakage (Phase 7G), 6. Recommended first experiment, SECTION 10 — Recommended calibration/evaluation split, SECTION 7 — Genuine/impostor trial feasibility, speaker_overlap(), split_trials_by_speaker(), summarize_trials(), Trial (+2 more)

### Community 18 - "package.json"
Cohesion: 0.07
Nodes (35): dependencies, @fontsource/tiro-devanagari-hindi@^5.3.0, @fontsource-variable/inter@^5.3.0, @fontsource-variable/source-serif-4@^5.3.1, react@^19.3.1, react-dom@^19.3.1, devDependencies, oxlint@^1.88.0 (+27 more)

### Community 19 - "Findings.tsx"
Cohesion: 0.13
Nodes (24): Dot, DotGrid(), dots, layout(), pile(), Placed, regimeFor(), SLICE_COLORS (+16 more)

### Community 20 - "check_head.py"
Cohesion: 0.19
Nodes (7): check(), free_port(), main(), wait_ready(), free_port(), test_ui_path_check_passes(), wait_until_ready()

### Community 21 - "run_indicsynth_finetune.py"
Cohesion: 0.11
Nodes (9): collect_fleurs_bonafide(), collect_indicsynth_spoofs(), compute_metrics(), evaluate(), extract_features(), load_n500_weights(), main(), preprocess_audio() (+1 more)

### Community 22 - "Analyzer.tsx"
Cohesion: 0.15
Nodes (24): Analyzer(), noLevel(), Props, describe(), SessionLog(), Analysis, analyze(), ModelState (+16 more)

### Community 23 - "Speaker Verification"
Cohesion: 0.20
Nodes (10): Architecture, ECAPA-TDNN speaker encoder, Encoder limitations, Hardware / device support, Manually inspecting a file (or a pair of files), Project layout, Relationship to the B.S. Detector pipeline, Running the encoder tests (+2 more)

### Community 24 - "calibration.py"
Cohesion: 0.18
Nodes (8): compute_eer(), compute_far_frr_curve(), compute_roc_auc(), _threshold_candidates(), ThresholdOperatingPoint, _to_float_list(), test_compute_eer_requires_both_classes(), test_compute_eer_returns_method_description()

### Community 25 - "Rules"
Cohesion: 0.09
Nodes (20): caveman, Example output, How to invoke, See also, What it does, 1. Answer first, 2. Kill ceremony, 3. Short word (+12 more)

### Community 26 - "App.tsx"
Cohesion: 0.14
Nodes (19): App(), initial, useReveal(), useStatus(), SampleRequest, Cursor(), Mode, CREDITS (+11 more)

### Community 27 - "train_sls_channel_robust.py"
Cohesion: 0.09
Nodes (14): apply_acoustic_channel(), apply_phone_speaker(), collect_fleurs_bonafide(), collect_indicsynth(), compute_metrics(), detect_active_speech(), evaluate(), extract_features() (+6 more)

### Community 28 - "test_datasets_trials.py"
Cohesion: 0.17
Nodes (19): sample_trials(), TrialValidationError, _make_trial(), _many_trials(), test_genuine_trial_requires_same_speaker(), test_impostor_trial_requires_different_speakers(), test_invalid_label_raises(), test_sample_trials_different_seeds_can_differ() (+11 more)

### Community 29 - "backend.py"
Cohesion: 0.20
Nodes (6): _l2n(), _np(), ScoreBackend, test_mean_removal_separates_speakers_sharing_a_channel_offset(), test_plain_backend_is_cosine(), test_round_trip_and_asnorm()

### Community 30 - "train_sls_robust.py"
Cohesion: 0.08
Nodes (14): edge(), main(), xtts(), collect_fleurs_bonafide(), collect_indicsynth(), compute_metrics(), detect_active_speech(), evaluate() (+6 more)

### Community 31 - "compilerOptions"
Cohesion: 0.10
Nodes (19): compilerOptions, allowArbitraryExtensions, allowImportingTsExtensions, erasableSyntaxOnly, jsx, lib, module, moduleDetection (+11 more)

### Community 32 - "VoiceInput.tsx"
Cohesion: 0.19
Nodes (17): axisTicks(), Props, Scope(), IconClose(), IconMic(), IconPlay(), IconStop(), IconUpload() (+9 more)

### Community 33 - "train_phase12.py"
Cohesion: 0.05
Nodes (34): main(), _h(), rank_split(), split_of(), main(), make_edge(), make_xtts(), ai4bharat() (+26 more)

### Community 34 - "app.py"
Cohesion: 0.14
Nodes (4): HashedAssets, lifespan(), load_all(), spa()

### Community 35 - "compute_far"
Cohesion: 0.14
Nodes (16): Architecture, Calibration & verification decision (Phase 6 — implemented), Cosine similarity (Phase 5 — implemented), Integration with B.S. Detector, Large-scale VoxCeleb calibration pipeline (Phase 7 — implemented), Out of scope for the baseline, Pipeline overview, Sections to be filled in as implementation proceeds (+8 more)

### Community 36 - "speaker-a-1"
Cohesion: 0.11
Nodes (17): speaker-a-1, pairs, results, analyze, recordedAt, samples, aggregation, device (+9 more)

### Community 37 - "sv_models.py"
Cohesion: 0.16
Nodes (6): Ecapa2, _hf(), ResNet293, ResNet34, WavLM, WeSpeaker

### Community 38 - "encoder.py"
Cohesion: 0.15
Nodes (10): ECAPA-TDNN speaker encoder (Phase 4 — implemented), Device handling, DeviceError, DeviceInfo, resolve_device(), test_encoder_initializes_on_cuda(), test_resolve_device_auto_returns_cpu_or_cuda(), test_resolve_device_explicit_cpu() (+2 more)

### Community 39 - "run_forensic_detection"
Cohesion: 0.24
Nodes (6): analyze(), detect(), detection_json(), verify(), ForensicInferenceResult, test_detection_json_sliding_window_structure()

### Community 40 - "compilerOptions"
Cohesion: 0.12
Nodes (16): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, lib, module, moduleDetection, noEmit, noFallthroughCasesInSwitch (+8 more)

### Community 41 - "Audio Deepfake Detection: SLS on Frozen XLS-R 300M (Hindi Audio Benchmark)"
Cohesion: 0.12
Nodes (15): 1. Executive Summary, 2.1 Dataset & Sampling, 2.2 Split Protocol & Disjointness Guarantees, 2.3 Model Architecture, 2.4 Training Hyperparameters (Few-Shot), 2. Frozen Experimental Protocol, 3.1 Consolidated Comparison Table, 3.2 Per-Seed Breakdown (+7 more)

### Community 42 - "verify"
Cohesion: 0.11
Nodes (18): run(), _all_segments(), _cos(), decide(), embed(), embed_second(), fused_score(), high_band_share() (+10 more)

### Community 43 - "verification"
Cohesion: 0.22
Nodes (16): ecapa, wespeaker, verification, verification, cosines, isCalibrated, latencyMs, margin (+8 more)

### Community 45 - "test_sliding_window.py"
Cohesion: 0.33
Nodes (7): extract_sliding_windows(), test_sliding_window_exact_length(), test_sliding_window_max_windows_cap(), test_sliding_window_multi_window_coverage(), test_sliding_window_onset_alignment_disabled(), test_sliding_window_short_audio(), test_sliding_window_with_leading_silence_onset_aligned()

### Community 46 - "inference.py"
Cohesion: 0.15
Nodes (8): find_speech_onset_index(), load_and_preprocess_audio(), pad_or_truncate_audio(), run_forensic_detection(), test_pad_or_truncate_audio_aligns_window(), test_pure_silence_rejection(), test_speech_onset_detection_with_leading_silence(), test_speech_onset_short_audio()

### Community 47 - "B.S-Detector: Parameter-Efficient Audio Deepfake Detection on Hindi Speech"
Cohesion: 0.13
Nodes (15): 1. Environment Setup, 2. Run CLI Inference, 3. Python API Usage, Architectural Specification, B.S-Detector: Parameter-Efficient Audio Deepfake Detection on Hindi Speech, Benchmark Dataset & Stratification, Checkpoint Manifest, Citation (+7 more)

### Community 48 - "InsufficientCalibrationDataError"
Cohesion: 0.15
Nodes (11): Current project status: calibration is NOT available, Our one real score: an uncalibrated example only, Threshold methodology (unchanged from Phase 6), Calibrating against real trial data, Calibration & verification decision, Running a verification decision, Running the calibration tests, CalibrationError (+3 more)

### Community 49 - "detection"
Cohesion: 0.36
Nodes (14): aggregation, device, durationSec, head, latencyMs, layerWeights, numWindows, probBonafide (+6 more)

### Community 50 - "referenceDetection"
Cohesion: 0.36
Nodes (14): referenceDetection, referenceDetection, aggregation, device, durationSec, head, latencyMs, layerWeights (+6 more)

### Community 51 - "Hero.tsx"
Cohesion: 0.25
Nodes (11): HeroArtifacts(), LayersArtifact(), pct(), SourceTag(), SpectrogramArtifact(), Stage1Artifact(), Stage2Artifact(), BANDS (+3 more)

### Community 52 - "B.S. Detector"
Cohesion: 0.20
Nodes (10): B.S. Detector, Credits, Developing the app, [▶ Download and run B.S. Detector](https://github.com/Anumay1231/B.S-Detector/archive/refs/heads/main.zip), How it works, Limitations, Repository layout, Reproducing the research (+2 more)

### Community 53 - "train_ecapa_indian.py"
Cohesion: 0.06
Nodes (33): 1. Kathbath, 2. HiACC, 3. IIITH Hindi-English Code-Mixed Speech Database, 4. Side-by-side comparison, 5. Final Hinglish decision, 6. Recommended final dataset strategy, 7. Remaining unknowns (require actual dataset/file access), 8. Sources (+25 more)

### Community 54 - "generate_figures.py"
Cohesion: 0.17
Nodes (8): main(), plot_adaptation_curve(), plot_confusion_matrices(), plot_dataset_stratification(), main(), plot_fig2_adaptation_curve(), plot_fig3_unfreezing(), plot_fig4_generator_stratification()

### Community 55 - "peakSpoofWindow"
Cohesion: 0.54
Nodes (13): peakSpoofWindow, peakSpoofWindow, end_sec, index, prob_bonafide, prob_spoof, start_sec, verdict (+5 more)

### Community 56 - "eval_ecapa_comparison.py"
Cohesion: 0.28
Nodes (8): apply_room(), compute_eer(), decode_audio_bytes(), evaluate_model(), get_speaker_split(), load_held_out_trials(), main(), make_room()

### Community 57 - "EncoderError"
Cohesion: 0.22
Nodes (9): Cross-lingual accuracy caveat, Expected input, Known limitation: model could not be downloaded in this sandbox, Model caching, Model identity and source, Phase 7 caveat: the model was itself trained on VoxCeleb, Pretrained model: speechbrain/spkrec-ecapa-voxceleb, Speaker verification API (not used) (+1 more)

### Community 58 - "fake-hindi-xtts"
Cohesion: 0.17
Nodes (12): fake-hindi-xtts, aggregation, device, durationSec, head, latencyMs, layerWeights, numWindows (+4 more)

### Community 59 - "real-hindi"
Cohesion: 0.17
Nodes (12): real-hindi, aggregation, device, durationSec, head, latencyMs, layerWeights, numWindows (+4 more)

### Community 60 - "speaker-a-2"
Cohesion: 0.17
Nodes (12): speaker-a-2, aggregation, device, durationSec, head, latencyMs, layerWeights, numWindows (+4 more)

### Community 61 - "speaker-b-1"
Cohesion: 0.17
Nodes (12): speaker-b-1, aggregation, device, durationSec, head, latencyMs, layerWeights, numWindows (+4 more)

### Community 62 - "json"
Cohesion: 0.23
Nodes (6): deepfake_clips(), main(), score(), speaker_clips(), write(), main()

### Community 63 - "deepfake_check.py"
Cohesion: 0.36
Nodes (6): Detector, kind_of(), lang_of(), main(), read(), _wav()

### Community 64 - "numpy"
Cohesion: 0.21
Nodes (14): extract_audio_array(), evaluate(), fit_constants(), fit_margin(), fold_of(), fuse(), load_set(), main() (+6 more)

### Community 65 - "different-speaker"
Cohesion: 0.18
Nodes (11): interlocked, interlockReason, interlocked, interlockReason, verification, verify, interlocked, interlockReason (+3 more)

### Community 66 - "Phase 7B — Dataset Discovery, Audit & Selection (English / Hindi / Hinglish)"
Cohesion: 0.11
Nodes (17): English, Hindi / Indian, Hinglish, Phase 7B — Dataset Discovery, Audit & Selection (English / Hindi / Hinglish), SECTION 11 — Expected disk/storage requirements, SECTION 12 — Why this combination is better than using one 40-language dataset, SECTION 13 — Risks/limitations, SECTION 14 — Exact next step (+9 more)

### Community 67 - "score_recordings.py"
Cohesion: 0.24
Nodes (4): decode(), load_people(), main(), pct()

### Community 68 - "build_trials.py"
Cohesion: 0.29
Nodes (6): build_subset(), _explain_missing_trial_list(), main(), test_build_subset_excludes_missing_files(), test_build_subset_respects_max_counts_and_splits(), _trial()

### Community 69 - "Agent Handoff — B.S-Detector (Audio Deepfake Detection)"
Cohesion: 0.22
Nodes (8): 1. What This Project Is, 2. Key Benchmark Findings ($N_{\text{test}} = 400$), 3. Dataset & Stratification Architecture, 4. Codebase Map, 5. Local Inference, 6. Phase 9: speaker check on Hindi / Indian English, and a detector caveat, Agent Handoff — B.S-Detector (Audio Deepfake Detection), Seed Highlights for $N=500$:

### Community 70 - "test_phone_band.py"
Cohesion: 0.18
Nodes (9): phone_line(), Ecapa, features(), Second, test_high_band_share_separates_wideband_from_a_phone_line(), test_phone_band_known_clip_is_named(), test_phone_band_test_clip_is_not_judged(), test_wideband_pair_is_judged() (+1 more)

### Community 71 - "ResultCard.tsx"
Cohesion: 0.31
Nodes (8): vitest@5.0.5, clamp01(), pct(), ResultCard(), stage1Words(), Stroke(), speakerMeter(), vitest

### Community 72 - "Recording protocol: real-world test set"
Cohesion: 0.25
Nodes (7): Consent template, File names, Recording protocol: real-world test set, Storage and privacy, What each person records, What to measure, Who and how many

### Community 73 - "server"
Cohesion: 0.25
Nodes (8): server, backbone, checkpoint, device, head, verifierModel, verifierSystem, windowSeconds

### Community 74 - "SLSHead"
Cohesion: 0.20
Nodes (4): load_detector_models(), select_best_device(), SLSHead, test_sls_head_batched_forward()

### Community 75 - "calibrate"
Cohesion: 0.16
Nodes (18): calibrate(), CalibrationResult, InvalidScoreError, test_all_scores_identical_value(), test_calibration_result_operating_points_cover_full_range(), test_calibration_result_thresholds_are_sorted_ascending(), test_far_is_non_increasing_with_threshold(), test_frr_is_non_decreasing_with_threshold() (+10 more)

### Community 77 - "Phase 9: Speaker verification on Hindi and Indian-English speakers"
Cohesion: 0.25
Nodes (8): 1. Data, 2. Root cause, 3. What we tried, and what each step bought, 4. Shipped decision, 5. Results (held out, 2-fold cross-validation over SVQ people), 6. Voice clones and the interlock, 7. The Indian-adapted ECAPA, tested fairly (30 Oct), Phase 9: Speaker verification on Hindi and Indian-English speakers

### Community 78 - ".oxlintrc.json"
Cohesion: 0.29
Nodes (6): ignorePatterns, plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 79 - "load_hindi_head"
Cohesion: 0.25
Nodes (3): load_hindi_head(), ModelState, status()

### Community 80 - "ui_path_check.py"
Cohesion: 0.23
Nodes (7): form(), main(), pairs(), post(), read(), trim(), wav()

### Community 81 - "Demo recordings"
Cohesion: 0.33
Nodes (5): 1. Real vs generated (`deepfake/`), 2. Same person or not (`speaker/`), Demo recordings, Sources and licences, Things to know before presenting

### Community 83 - "Stage 1 (deepfake detection): measured gaps and improvement plan"
Cohesion: 0.29
Nodes (6): Phase 12 head (the app's default since 30 Oct), Plan, in order of payoff, Silence-invariant head (the app's default since 28 Oct), Stage 1 (deepfake detection): measured gaps and improvement plan, The head leans on leading silence (the most important finding), What we measured (October 2026)

### Community 84 - "Sample audio credits"
Cohesion: 0.33
Nodes (5): fake-hindi-xtts.wav: synthetic speech, real-hindi.wav: Google FLEURS, Sample audio credits, speaker-a-1.wav, speaker-a-2.wav: LibriSpeech, speaker-b-1.wav: Google SVQ

### Community 85 - "check_environment.py"
Cohesion: 0.33
Nodes (5): check_cuda(), check_tensor_ops(), check_versions(), main(), _print_header()

### Community 86 - "Phase 7D — Dataset Access & Final Verification"
Cohesion: 0.17
Nodes (11): 1. Kathbath, 2. IIITH Hindi-English Code-Mixed Database, 3. VoxCeleb1, 4. HiACC, 5. Final dataset decision, 7. Storage estimate, 8. Manual actions required from me, Applying Phase 7D's explicit file-structure rule (§3 of your request) (+3 more)

### Community 90 - "metrics.py"
Cohesion: 0.15
Nodes (4): compute_all_metrics(), compute_eer(), print_comparison_table(), print_metrics()

### Community 98 - "load_speaker_name_map"
Cohesion: 0.18
Nodes (12): load_speaker_name_map(), SpeakerMetadataError, test_load_speaker_name_map_accepts_whitespace_separated(), test_load_speaker_name_map_conflicting_ids_raises(), test_load_speaker_name_map_missing_file_raises(), test_load_speaker_name_map_no_usable_rows_raises(), test_load_speaker_name_map_parses_official_format(), test_load_speaker_name_map_skips_header_and_blank_lines() (+4 more)

### Community 99 - "Calibration and the verification decision (Phase 6)"
Cohesion: 0.15
Nodes (13): API summary, Calibration and the verification decision (Phase 6), Data requirement for real calibration, Equal Error Rate (EER), Explicitly out of scope for Phase 6, How to actually calibrate this system, How to read these numbers, Phase 8B — First real calibration results (VoxCeleb1) (+5 more)

### Community 100 - "scripts/test_encoder.py"
Cohesion: 0.31
Nodes (4): main(), _report_single(), _report_two_file(), timed_encode()

### Community 101 - "test_app_speaker.py"
Cohesion: 0.21
Nodes (7): test_clear_voice_has_high_snr_and_keeps_speech(), test_continuous_speech_is_not_trimmed_away(), test_long_clips_are_embedded_as_the_mean_of_3_s_segments(), test_noise_only_clip_is_refused_before_the_encoder(), test_threshold_rises_with_speech_and_is_clamped(), test_too_little_speech_is_refused(), voice_in_room()

### Community 103 - "compute_score_statistics"
Cohesion: 0.29
Nodes (5): compute_score_statistics(), ScoreStatistics, test_score_statistics_accepts_tensor(), test_score_statistics_basic(), test_score_statistics_single_value_has_zero_std()

### Community 104 - "Phase 7: large-scale calibration using VoxCeleb"
Cohesion: 0.20
Nodes (10): Deterministic subset sampling (Phase 7B), Domain limitation: VoxCeleb vs. Hindi/Hinglish (Phase 7M), Embedding cache design, Explicitly out of scope for Phase 7, Personal recordings stay out of VoxCeleb calibration (Phase 7L), Phase 7: large-scale calibration using VoxCeleb, Pipeline, Trial labels (+2 more)

### Community 105 - "Phase 8C — Full VoxCeleb1 official-protocol benchmark"
Cohesion: 0.33
Nodes (6): Phase 8C — Full VoxCeleb1 official-protocol benchmark, Results, The Phase 8B threshold, measured on the full protocol, Trial-list formats: numeric ids vs. speaker names, Why 1.04% and not the published ~0.80%, Why the full protocol scores *worse* than the speaker-disjoint split

### Community 109 - "Large-scale VoxCeleb calibration (Phase 7)"
Cohesion: 0.50
Nodes (4): Large-scale VoxCeleb calibration (Phase 7), Personal recordings stay separate, Running the Phase 7 tests, Running the pipeline on real data

### Community 110 - "Results"
Cohesion: 0.50
Nodes (4): Results, Stage 1: deepfake detection on Hindi, Stage 2: speaker verification on Hindi and Indian English (Phase 9), Stage 2: speaker verification on VoxCeleb1

### Community 112 - "Setup"
Cohesion: 0.40
Nodes (5): Create and activate a virtual environment, Install dependencies, Requirements, Setup, Verify the environment

## Knowledge Gaps
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SpeakerEncoder` connect `SpeakerEncoder` to `pytest`, `VerifierState`, `EmbeddingCache`, `scripts/test_encoder.py`, `encoder.py`, `encoder.py`, `Ecapa`, `verify`, `InsufficientCalibrationDataError`, `Speaker Verification`, `EncoderError`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Should `pytest` be split into smaller, more focused modules?**
  _Cohesion score 0.05862939235099981 - nodes in this community are weakly interconnected._
- **Why does `calibrate()` connect `calibrate` to `Calibration and the verification decision (Phase 6)`, `compute_far`, `compute_score_statistics`, `encoder.py`, `Phase 7: large-scale calibration using VoxCeleb`, `InsufficientCalibrationDataError`, `run_calibration`, `calibration.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Should `experiment.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._
- **Why does `Trial` connect `Trial` to `EmbeddingCache`, `Phase 7B — Dataset Discovery, Audit & Selection (English / Hindi / Hinglish)`, `compute_far`, `build_trials.py`, `test_datasets_voxceleb.py`, `Phase 7D — Dataset Access & Final Verification`, `test_datasets_trials.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Should `EmbeddingCache` be split into smaller, more focused modules?**
  _Cohesion score 0.05040611562350693 - nodes in this community are weakly interconnected._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.10668563300142248 - nodes in this community are weakly interconnected._