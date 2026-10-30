# Graph Report - B.S-Detector  (2026-10-30)

## Corpus Check
- 144 files · ~116,765 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 38 file(s) not represented in the graph (top: (none) 15, .css 14, .pth 4)

## Summary
- 1824 nodes · 4019 edges · 97 communities (81 shown, 16 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 404 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0d2f404a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pytest
- typing
- EmbeddingCache
- pathlib
- Dataset Auditing & Subset Generation
- audio.py
- run.py
- speaker.py
- test_datasets_voxceleb.py
- Steep — Style Reference
- test_multi_reference.py
- audio.ts
- numpy
- SpeakerEncoder
- sv_benchmark.py
- run_calibration
- trials.py
- package.json
- Findings.tsx
- encoder.py
- run_indicsynth_finetune.py
- Analyzer.tsx
- Speaker Verification
- calibrate
- Rules
- App.tsx
- train_sls_channel_robust.py
- test_datasets_trials.py
- backend.py
- train_sls_robust.py
- compilerOptions
- generate_figures.py
- datasets/__init__.py
- app.py
- compute_far
- speaker-a-1
- sv_models.py
- resolve_device
- ResultCard.tsx
- compilerOptions
- Audio Deepfake Detection: SLS on Frozen XLS-R 300M (Hindi Audio Benchmark)
- react
- verification
- calibration.py
- test_sliding_window.py
- test_inference_silence.py
- B.S-Detector: Parameter-Efficient Audio Deepfake Detection on Hindi Speech
- InsufficientCalibrationDataError
- detection
- referenceDetection
- Hero.tsx
- B.S. Detector
- train_ecapa_indian.py
- Calibration and the verification decision (Phase 6)
- peakSpoofWindow
- eval_ecapa_comparison.py
- extract_slices_from_raw
- fake-hindi-xtts
- real-hindi
- speaker-a-2
- speaker-b-1
- json
- deepfake_check.py
- collect_validation_pairs
- different-speaker
- detection_json
- check_environment.py
- Phase 7: large-scale calibration using VoxCeleb
- Agent Handoff — B.S-Detector (Audio Deepfake Detection)
- Pretrained model: speechbrain/spkrec-ecapa-voxceleb
- scripts/test_encoder.py
- Recording protocol: real-world test set
- server
- inference.py
- SLSHead
- Phase 9: Speaker verification on Hindi and Indian-English speakers
- .oxlintrc.json
- load_hindi_head
- compute_score_statistics
- Demo recordings
- Phase 8C — Full VoxCeleb1 official-protocol benchmark
- Stage 1 (deepfake detection): measured gaps and improvement plan
- Sample audio credits
- make_fakes.py
- evaluate
- demo/manifest.json
- tsconfig.json
- generate_edge_tts_samples
- AGENTS.md

## God Nodes (most connected - your core abstractions)
1. `SpeakerEncoder` - 45 edges
2. `cosine_similarity()` - 41 edges
3. `calibrate()` - 38 edges
4. `EmbeddingCache` - 37 edges
5. `SpeakerVerifier` - 30 edges
6. `parse_trial_list()` - 25 edges
7. `run_calibration()` - 24 edges
8. `_StubEncoder` - 22 edges
9. `preprocess_audio()` - 21 edges
10. `f()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `1. Data` --references--> `train()`  [INFERRED]
  docs/speaker-verification-hindi-english.md → scripts/train_ecapa_indian.py
- `Deterministic subset sampling (Phase 7B)` --references--> `sample_trials()`  [INFERRED]
  docs/calibration.md → src/speaker_verification/datasets/trials.py
- `Hardware / device support` --references--> `resolve_device()`  [INFERRED]
  docs/speaker-verification.md → src/speaker_verification/encoder.py
- `Model caching` --references--> `SpeakerEncoder`  [INFERRED]
  docs/model.md → src/speaker_verification/encoder.py
- `Real ECAPA integration test (two audio files)` --references--> `SpeakerEncoder`  [INFERRED]
  docs/speaker-verification.md → src/speaker_verification/encoder.py

## Import Cycles
- None detected.

## Communities (97 total, 16 thin omitted)

### Community 0 - "pytest"
Cohesion: 0.06
Nodes (55): cosine_similarity(), EmbeddingMismatchError, InvalidEmbeddingError, SimilarityError, _validate_embedding(), InvalidThresholdError, SpeakerVerifier, _validate_threshold() (+47 more)

### Community 1 - "typing"
Cohesion: 0.05
Nodes (22): evaluate_zeroshot(), run_fewshot_experiment(), _run_inference(), train_fewshot(), collate_features(), extract_and_cache_features(), load_cached_features(), load_features_batch() (+14 more)

### Community 2 - "EmbeddingCache"
Cohesion: 0.05
Nodes (35): write_trials_csv(), extract_embeddings(), f(), main(), _read_trials_csv(), score_trials(), _write_scores_csv(), EmbeddingCache (+27 more)

### Community 3 - "pathlib"
Cohesion: 0.09
Nodes (10): collect_unique_utterances(), main(), print_comparison_table(), main(), create_fewshot_pools(), create_test_split(), get_all_needed_ids(), get_labels_for_ids() (+2 more)

### Community 4 - "Dataset Auditing & Subset Generation"
Cohesion: 0.12
Nodes (6): extract_audio_array(), collect_hindi_metadata(), main(), print_diagnostics(), stratified_sample(), generate_metadata_parquet()

### Community 5 - "audio.py"
Cohesion: 0.09
Nodes (32): Audio preprocessing (Phase 3 — implemented), Audio preprocessing, Manually inspecting a single file, Running the audio preprocessing tests, main(), AudioDecodeError, AudioError, AudioFileNotFoundError (+24 more)

### Community 6 - "run.py"
Cohesion: 0.07
Nodes (23): ensure_environment(), ensure_frontend(), ensure_hindi_head(), main(), open_when_up(), port_free(), say(), venv_python() (+15 more)

### Community 8 - "speaker.py"
Cohesion: 0.07
Nodes (25): run(), _all_segments(), _cos(), decide(), embed(), embed_second(), fused_score(), _log_d() (+17 more)

### Community 9 - "test_datasets_voxceleb.py"
Cohesion: 0.10
Nodes (31): normalize_utterance_id(), parse_trial_list(), speaker_id_from_utterance_id(), TrialListParseError, test_load_speaker_name_map_parses_official_format(), test_load_speaker_name_map_skips_header_and_blank_lines(), test_load_speaker_name_map_strips_padded_fields(), test_normalize_backslash_paths_are_handled() (+23 more)

### Community 10 - "Steep — Style Reference"
Cohesion: 0.05
Nodes (37): Accent Peach Card, Agent Prompt Guide, Avatar Bubble, Border Radius, Components, CSS Custom Properties, Do, Do's and Don'ts (+29 more)

### Community 11 - "test_multi_reference.py"
Cohesion: 0.09
Nodes (19): test_index_html_is_revalidated(), client(), Ecapa, features(), post(), Second, test_a_bad_extra_clip_is_named(), test_more_than_three_clips_are_refused() (+11 more)

### Community 12 - "audio.ts"
Cohesion: 0.11
Nodes (30): axisTicks(), Props, Scope(), IconClose(), IconMic(), IconPlay(), IconStop(), IconUpload() (+22 more)

### Community 13 - "numpy"
Cohesion: 0.14
Nodes (22): embed(), fit(), fit_threshold(), groups(), load(), main(), pair_trials(), report() (+14 more)

### Community 14 - "SpeakerEncoder"
Cohesion: 0.13
Nodes (19): Ecapa, InvalidWaveformError, SpeakerEncoder, _make_sine_wave(), speaker_encoder(), test_embedding_has_expected_dimension(), test_embedding_is_finite(), test_encode_accepts_1d_waveform() (+11 more)

### Community 15 - "sv_benchmark.py"
Cohesion: 0.12
Nodes (21): Backend, corpora(), _cv_files(), cv_speakers(), _decode(), eer(), embed(), evaluate() (+13 more)

### Community 16 - "run_calibration"
Cohesion: 0.14
Nodes (22): _format_report(), main(), _median(), _read_scores(), run_calibration(), _speakers(), _utterances(), _row() (+14 more)

### Community 17 - "trials.py"
Cohesion: 0.11
Nodes (15): Speaker leakage (Phase 7G), build_subset(), _explain_missing_trial_list(), main(), speaker_overlap(), split_trials_by_speaker(), summarize_trials(), Trial (+7 more)

### Community 18 - "package.json"
Cohesion: 0.09
Nodes (27): devDependencies, oxlint@^1.88.0, @types/node@^24.19.2, @types/react@^19.3.0, @types/react-dom@^19.3.0, typescript@~6.0.3, vite@^8.3.4, @vitejs/plugin-react@^6.1.2 (+19 more)

### Community 19 - "Findings.tsx"
Cohesion: 0.13
Nodes (24): Dot, DotGrid(), dots, layout(), pile(), Placed, regimeFor(), SLICE_COLORS (+16 more)

### Community 20 - "encoder.py"
Cohesion: 0.12
Nodes (10): Embedding output, _explain_missing_trials_file(), main(), _read_trials(), main(), _encode(), main(), main() (+2 more)

### Community 21 - "run_indicsynth_finetune.py"
Cohesion: 0.11
Nodes (9): collect_fleurs_bonafide(), collect_indicsynth_spoofs(), compute_metrics(), evaluate(), extract_features(), load_n500_weights(), main(), preprocess_audio() (+1 more)

### Community 22 - "Analyzer.tsx"
Cohesion: 0.22
Nodes (20): Analyzer(), noLevel(), Props, Analysis, analyze(), ModelState, post(), Status (+12 more)

### Community 23 - "Speaker Verification"
Cohesion: 0.09
Nodes (23): Architecture, Cosine similarity, Create and activate a virtual environment, ECAPA-TDNN speaker encoder, Encoder limitations, Hardware / device support, Install dependencies, Large-scale VoxCeleb calibration (Phase 7) (+15 more)

### Community 24 - "calibrate"
Cohesion: 0.16
Nodes (18): calibrate(), CalibrationResult, InvalidScoreError, test_all_scores_identical_value(), test_calibration_result_operating_points_cover_full_range(), test_calibration_result_thresholds_are_sorted_ascending(), test_far_is_non_increasing_with_threshold(), test_frr_is_non_decreasing_with_threshold() (+10 more)

### Community 25 - "Rules"
Cohesion: 0.09
Nodes (20): caveman, Example output, How to invoke, See also, What it does, 1. Answer first, 2. Kill ceremony, 3. Short word (+12 more)

### Community 26 - "App.tsx"
Cohesion: 0.16
Nodes (17): App(), initial, useReveal(), useStatus(), SampleRequest, Cursor(), CREDITS, Footer() (+9 more)

### Community 27 - "train_sls_channel_robust.py"
Cohesion: 0.14
Nodes (8): collect_fleurs_bonafide(), collect_indicsynth(), compute_metrics(), evaluate(), extract_features(), load_checkpoint_weights(), SLSHead, train_pipeline()

### Community 28 - "test_datasets_trials.py"
Cohesion: 0.19
Nodes (17): sample_trials(), TrialValidationError, _make_trial(), _many_trials(), test_genuine_trial_requires_same_speaker(), test_impostor_trial_requires_different_speakers(), test_invalid_label_raises(), test_sample_trials_different_seeds_can_differ() (+9 more)

### Community 29 - "backend.py"
Cohesion: 0.18
Nodes (6): _l2n(), _np(), ScoreBackend, test_mean_removal_separates_speakers_sharing_a_channel_offset(), test_plain_backend_is_cosine(), test_round_trip_and_asnorm()

### Community 30 - "train_sls_robust.py"
Cohesion: 0.15
Nodes (7): collect_fleurs_bonafide(), collect_indicsynth(), detect_active_speech(), extract_features(), extract_slices_from_raw(), generate_edge_tts_samples(), train_pipeline()

### Community 31 - "compilerOptions"
Cohesion: 0.10
Nodes (19): compilerOptions, allowArbitraryExtensions, allowImportingTsExtensions, erasableSyntaxOnly, jsx, lib, module, moduleDetection (+11 more)

### Community 32 - "generate_figures.py"
Cohesion: 0.17
Nodes (8): main(), plot_adaptation_curve(), plot_confusion_matrices(), plot_dataset_stratification(), main(), plot_fig2_adaptation_curve(), plot_fig3_unfreezing(), plot_fig4_generator_stratification()

### Community 33 - "datasets/__init__.py"
Cohesion: 0.16
Nodes (10): DatasetError, load_speaker_name_map(), resolve_utterance_path(), SpeakerMetadataError, test_load_speaker_name_map_accepts_whitespace_separated(), test_load_speaker_name_map_conflicting_ids_raises(), test_load_speaker_name_map_missing_file_raises(), test_load_speaker_name_map_no_usable_rows_raises() (+2 more)

### Community 34 - "app.py"
Cohesion: 0.13
Nodes (5): HashedAssets, lifespan(), load_all(), spa(), status()

### Community 35 - "compute_far"
Cohesion: 0.14
Nodes (16): Architecture, Calibration & verification decision (Phase 6 — implemented), Cosine similarity (Phase 5 — implemented), Integration with B.S. Detector, Large-scale VoxCeleb calibration pipeline (Phase 7 — implemented), Out of scope for the baseline, Pipeline overview, Sections to be filled in as implementation proceeds (+8 more)

### Community 36 - "speaker-a-1"
Cohesion: 0.11
Nodes (17): speaker-a-1, pairs, results, analyze, recordedAt, samples, aggregation, device (+9 more)

### Community 37 - "sv_models.py"
Cohesion: 0.16
Nodes (6): Ecapa2, _hf(), ResNet293, ResNet34, WavLM, WeSpeaker

### Community 38 - "resolve_device"
Cohesion: 0.15
Nodes (10): ECAPA-TDNN speaker encoder (Phase 4 — implemented), Device handling, DeviceError, DeviceInfo, resolve_device(), test_encoder_initializes_on_cuda(), test_resolve_device_auto_returns_cpu_or_cuda(), test_resolve_device_explicit_cpu() (+2 more)

### Community 39 - "ResultCard.tsx"
Cohesion: 0.21
Nodes (14): clamp01(), pct(), ResultCard(), stage1Words(), Stroke(), describe(), SessionLog(), speakerMeter() (+6 more)

### Community 40 - "compilerOptions"
Cohesion: 0.12
Nodes (16): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, lib, module, moduleDetection, noEmit, noFallthroughCasesInSwitch (+8 more)

### Community 41 - "Audio Deepfake Detection: SLS on Frozen XLS-R 300M (Hindi Audio Benchmark)"
Cohesion: 0.12
Nodes (15): 1. Executive Summary, 2.1 Dataset & Sampling, 2.2 Split Protocol & Disjointness Guarantees, 2.3 Model Architecture, 2.4 Training Hyperparameters (Few-Shot), 2. Frozen Experimental Protocol, 3.1 Consolidated Comparison Table, 3.2 Per-Seed Breakdown (+7 more)

### Community 42 - "react"
Cohesion: 0.17
Nodes (12): dependencies, @fontsource/tiro-devanagari-hindi@^5.3.0, @fontsource-variable/inter@^5.3.0, @fontsource-variable/source-serif-4@^5.3.1, react@^19.3.1, react-dom@^19.3.1, Mode, @fontsource/tiro-devanagari-hindi (+4 more)

### Community 43 - "verification"
Cohesion: 0.22
Nodes (16): ecapa, wespeaker, verification, verification, cosines, isCalibrated, latencyMs, margin (+8 more)

### Community 44 - "calibration.py"
Cohesion: 0.18
Nodes (8): compute_eer(), compute_far_frr_curve(), compute_roc_auc(), _threshold_candidates(), ThresholdOperatingPoint, _to_float_list(), test_compute_eer_requires_both_classes(), test_compute_eer_returns_method_description()

### Community 45 - "test_sliding_window.py"
Cohesion: 0.18
Nodes (9): extract_sliding_windows(), SLSHead, test_sliding_window_exact_length(), test_sliding_window_max_windows_cap(), test_sliding_window_multi_window_coverage(), test_sliding_window_onset_alignment_disabled(), test_sliding_window_short_audio(), test_sliding_window_with_leading_silence_onset_aligned() (+1 more)

### Community 46 - "test_inference_silence.py"
Cohesion: 0.17
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
Cohesion: 0.14
Nodes (14): B.S. Detector, Credits, Developing the app, [▶ Download and run B.S. Detector](https://github.com/Anumay1231/B.S-Detector/archive/refs/heads/main.zip), How it works, Limitations, Repository layout, Reproducing the research (+6 more)

### Community 53 - "train_ecapa_indian.py"
Cohesion: 0.20
Nodes (6): AAMSoftmax, collect_training_samples(), ensure_datasets_downloaded(), evaluate_pairs(), get_speaker_split(), train()

### Community 54 - "Calibration and the verification decision (Phase 6)"
Cohesion: 0.15
Nodes (13): API summary, Calibration and the verification decision (Phase 6), Data requirement for real calibration, Equal Error Rate (EER), Explicitly out of scope for Phase 6, How to actually calibrate this system, How to read these numbers, Phase 8B — First real calibration results (VoxCeleb1) (+5 more)

### Community 55 - "peakSpoofWindow"
Cohesion: 0.54
Nodes (13): peakSpoofWindow, peakSpoofWindow, end_sec, index, prob_bonafide, prob_spoof, start_sec, verdict (+5 more)

### Community 56 - "eval_ecapa_comparison.py"
Cohesion: 0.28
Nodes (8): apply_room(), compute_eer(), decode_audio_bytes(), evaluate_model(), get_speaker_split(), load_held_out_trials(), main(), make_room()

### Community 57 - "extract_slices_from_raw"
Cohesion: 0.19
Nodes (5): apply_acoustic_channel(), apply_phone_speaker(), detect_active_speech(), extract_slices_from_raw(), make_room_impulse()

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
Cohesion: 0.30
Nodes (5): deepfake_clips(), main(), score(), speaker_clips(), write()

### Community 63 - "deepfake_check.py"
Cohesion: 0.32
Nodes (6): Detector, kind_of(), lang_of(), main(), read(), _wav()

### Community 64 - "collect_validation_pairs"
Cohesion: 0.25
Nodes (5): apply_room(), collect_validation_pairs(), decode_audio_bytes(), IndianSpeakerDataset, make_room()

### Community 65 - "different-speaker"
Cohesion: 0.18
Nodes (11): interlocked, interlockReason, interlocked, interlockReason, verification, verify, interlocked, interlockReason (+3 more)

### Community 66 - "detection_json"
Cohesion: 0.24
Nodes (6): analyze(), detect(), detection_json(), verify(), ForensicInferenceResult, test_detection_json_sliding_window_structure()

### Community 67 - "check_environment.py"
Cohesion: 0.33
Nodes (5): check_cuda(), check_tensor_ops(), check_versions(), main(), _print_header()

### Community 68 - "Phase 7: large-scale calibration using VoxCeleb"
Cohesion: 0.20
Nodes (10): Deterministic subset sampling (Phase 7B), Domain limitation: VoxCeleb vs. Hindi/Hinglish (Phase 7M), Embedding cache design, Explicitly out of scope for Phase 7, Personal recordings stay out of VoxCeleb calibration (Phase 7L), Phase 7: large-scale calibration using VoxCeleb, Pipeline, Trial labels (+2 more)

### Community 69 - "Agent Handoff — B.S-Detector (Audio Deepfake Detection)"
Cohesion: 0.22
Nodes (8): 1. What This Project Is, 2. Key Benchmark Findings ($N_{\text{test}} = 400$), 3. Dataset & Stratification Architecture, 4. Codebase Map, 5. Local Inference, 6. Phase 9: speaker check on Hindi / Indian English, and a detector caveat, Agent Handoff — B.S-Detector (Audio Deepfake Detection), Seed Highlights for $N=500$:

### Community 70 - "Pretrained model: speechbrain/spkrec-ecapa-voxceleb"
Cohesion: 0.22
Nodes (9): Cross-lingual accuracy caveat, Expected input, Known limitation: model could not be downloaded in this sandbox, Model caching, Model identity and source, Phase 7 caveat: the model was itself trained on VoxCeleb, Pretrained model: speechbrain/spkrec-ecapa-voxceleb, Speaker verification API (not used) (+1 more)

### Community 71 - "scripts/test_encoder.py"
Cohesion: 0.31
Nodes (4): main(), _report_single(), _report_two_file(), timed_encode()

### Community 72 - "Recording protocol: real-world test set"
Cohesion: 0.25
Nodes (7): Consent template, File names, Recording protocol: real-world test set, Storage and privacy, What each person records, What to measure, Who and how many

### Community 73 - "server"
Cohesion: 0.25
Nodes (8): server, backbone, checkpoint, device, head, verifierModel, verifierSystem, windowSeconds

### Community 77 - "Phase 9: Speaker verification on Hindi and Indian-English speakers"
Cohesion: 0.29
Nodes (7): 1. Data, 2. Root cause, 3. What we tried, and what each step bought, 4. Shipped decision, 5. Results (held out, 2-fold cross-validation over SVQ people), 6. Voice clones and the interlock, Phase 9: Speaker verification on Hindi and Indian-English speakers

### Community 78 - ".oxlintrc.json"
Cohesion: 0.29
Nodes (6): ignorePatterns, plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 80 - "compute_score_statistics"
Cohesion: 0.29
Nodes (5): compute_score_statistics(), ScoreStatistics, test_score_statistics_accepts_tensor(), test_score_statistics_basic(), test_score_statistics_single_value_has_zero_std()

### Community 81 - "Demo recordings"
Cohesion: 0.33
Nodes (5): 1. Real vs generated (`deepfake/`), 2. Same person or not (`speaker/`), Demo recordings, Sources and licences, Things to know before presenting

### Community 82 - "Phase 8C — Full VoxCeleb1 official-protocol benchmark"
Cohesion: 0.33
Nodes (6): Phase 8C — Full VoxCeleb1 official-protocol benchmark, Results, The Phase 8B threshold, measured on the full protocol, Trial-list formats: numeric ids vs. speaker names, Why 1.04% and not the published ~0.80%, Why the full protocol scores *worse* than the speaker-disjoint split

### Community 83 - "Stage 1 (deepfake detection): measured gaps and improvement plan"
Cohesion: 0.33
Nodes (5): Plan, in order of payoff, Silence-invariant head (the app's default since 28 Oct), Stage 1 (deepfake detection): measured gaps and improvement plan, The head leans on leading silence (the most important finding), What we measured (October 2026)

### Community 84 - "Sample audio credits"
Cohesion: 0.33
Nodes (5): fake-hindi-xtts.wav: synthetic speech, real-hindi.wav: Google FLEURS, Sample audio credits, speaker-a-1.wav, speaker-a-2.wav: LibriSpeech, speaker-b-1.wav: Google SVQ

### Community 85 - "make_fakes.py"
Cohesion: 0.53
Nodes (3): edge(), main(), xtts()

## Knowledge Gaps
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `calibrate()` connect `calibrate` to `compute_far`, `Phase 7: large-scale calibration using VoxCeleb`, `calibration.py`, `InsufficientCalibrationDataError`, `run_calibration`, `compute_score_statistics`, `encoder.py`, `Calibration and the verification decision (Phase 6)`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Are the 35 inferred relationships involving `SpeakerEncoder` (e.g. with `ECAPA-TDNN speaker encoder (Phase 4 — implemented)` and `Our one real score: an uncalibrated example only`) actually correct?**
  _`SpeakerEncoder` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Should `pytest` be split into smaller, more focused modules?**
  _Cohesion score 0.06314432989690721 - nodes in this community are weakly interconnected._
- **Why does `SpeakerEncoder` connect `SpeakerEncoder` to `pytest`, `pathlib`, `Pretrained model: speechbrain/spkrec-ecapa-voxceleb`, `resolve_device`, `scripts/test_encoder.py`, `speaker.py`, `InsufficientCalibrationDataError`, `encoder.py`, `Speaker Verification`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `typing` be split into smaller, more focused modules?**
  _Cohesion score 0.0502283105022831 - nodes in this community are weakly interconnected._
- **Why does `Phase 7: large-scale calibration using VoxCeleb` connect `Phase 7: large-scale calibration using VoxCeleb` to `compute_far`, `InsufficientCalibrationDataError`, `trials.py`, `Calibration and the verification decision (Phase 6)`, `calibrate`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Should `EmbeddingCache` be split into smaller, more focused modules?**
  _Cohesion score 0.05185779203421545 - nodes in this community are weakly interconnected._