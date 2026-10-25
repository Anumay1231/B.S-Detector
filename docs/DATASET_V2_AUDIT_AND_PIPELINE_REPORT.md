# SEA-Spoof Dataset V2 Audit, Balanced Benchmark & Pipeline Report

**Date**: 2026-09-24  
**Project**: B.S-Detector (Audio Deepfake Detection for Hindi Speech via SLS on XLS-R 300M)  
**Status**: V2 Dataset Created & Verified | Pipeline Upgraded | Knowledge Graph Integrated  

---

## 1. Executive Summary

This report documents the comprehensive audit of the **SEA-Spoof** dataset, the discovery of severe generator bias in the initial V1 Hindi evaluation split, the construction of the **V2 stratified benchmark dataset**, and the technical upgrades made to the audio streaming and feature extraction pipeline.

### Key Milestones Achieved:
1. **Full-Corpus Dataset Audit**: Analyzed the ~439,362 row SEA-Spoof dataset (`Jack-ppkdczgx/SEA-Spoof`) to uncover the distribution of Hindi bonafide and spoof utterances.
2. **Dense Scanning & Offset Clustering**: Identified narrow cluster bands for bonafide speech (offsets ~35,148–35,247 and ~43,935–44,034) and mined a total pool of **15,882 bonafide Hindi clips** and **500+ spoof clips** across 5 distinct TTS models.
3. **Creation of Balanced V2 Dataset (`sea_spoof_en_hi_metadata_v2.parquet`)**:
   - Total samples: **800** (400 bonafide, 400 spoof).
   - **Spoof stratification**: Exactly **20.0% (80 clips)** from each of the 5 Hindi TTS architectures: `indic_tts`, `edge_tts`, `xtts-v2`, `vits_mms`, and `elevenlabs`.
   - **Bonafide multi-source**: 245 clips (61.2%) from IndicTTS and 155 clips (38.8%) from Mozilla CommonVoice.
   - Drastic reduction of single-source bias (only 89/800 rows overlap with V1).
4. **Streaming & Decoding Engine Overhaul**:
   - Resolved Windows `libtorchcodec` DLL load failures by bypassing HF audio decoding (`Audio(decode=False)`) and streaming raw byte payloads into in-memory `soundfile` decoders.
5. **Architectural Knowledge Graph (`graphify`)**:
   - Generated full AST-level semantic knowledge graph with 125 nodes, 242 edges, and 8 communities documented in `graphify-out/`.

---

## 2. Dataset Audit: V1 Bias vs. V2 Balanced Design

### 2.1 The Problem in V1 (`sea_spoof_en_hi_metadata.parquet`)
In the original V1 setup:
- 400 bonafide clips were taken from train shard 2, row groups 2–3 (solely IndicTTS).
- 400 spoof clips were taken from train shard 0, row groups 0–1, heavily skewed toward a single model (`hindi_indic_indic-tts`).
- Crucial state-of-the-art neural TTS architectures such as ElevenLabs, XTTS-v2, and Edge-TTS were severely underrepresented or absent.
- Models trained or evaluated on V1 learned artifact shortcuts specific to a single synthesizer rather than generalizable acoustic spoofing patterns.

### 2.2 Systematic Scanning & Discovery
Using `scripts/audit_sea_spoof.py`, `scripts/generate_corrected_hindi_subset.py`, and `scripts/_test_streaming.py`:
- **Phase 1 (Sparse Probe)**: Probed 50 evenly spaced intervals across the 439k row dataset, collecting 700 Hindi rows (500 spoof, 200 bonafide).
- **Phase 2 (Dense Scanning)**: Formed 19 targeted scan regions around bonafide-rich offset bands (~20,686 rows examined).
- **Discovery**: Uncovered 15,882 bonafide rows and confirmed 500 spoof rows evenly distributed across 5 TTS generators.

### 2.3 Dataset V1 vs. V2 Comparison

| Metric / Dimension | V1 Metadata (`sea_spoof_en_hi_metadata.parquet`) | V2 Metadata (`sea_spoof_en_hi_metadata_v2.parquet`) |
| :--- | :--- | :--- |
| **Total Clips** | 800 (400 spoof, 400 bonafide) | 800 (400 spoof, 400 bonafide) |
| **TTS Models Represented** | Heavily skewed to 1–2 generators | **5 Models Equally Stratified (20% each)** |
| **`hindi_indic_indic-tts`** | >50% | **80 clips (20.0%)** |
| **`hindi_common_edge_tts`** | Minimal / omitted | **80 clips (20.0%)** |
| **`hindi_indic_tts_xtts-v2`**| Minimal / omitted | **80 clips (20.0%)** |
| **`hindi_common_vits_mms`**  | Minimal / omitted | **80 clips (20.0%)** |
| **`elevenlabs`**             | Minimal / omitted | **80 clips (20.0%)** |
| **Bonafide Diversity** | IndicTTS single source | **IndicTTS (61.2%) + CommonVoice (38.8%)** |
| **V1 Overlap** | N/A (Original) | **89 / 800 rows (11.1%)** |

---

## 3. Engineering Fixes & Pipeline Enhancements

### 3.1 Resolving the Windows `torchcodec` Exception
During HuggingFace dataset streaming on Windows, `datasets.IterableDataset` defaults to `torchcodec` for audio decoding. Because Windows systems often lack the specific `libtorchcodec_core*.dll` compiled for system-shared FFmpeg, streaming previously aborted with:
```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment...
  OSError: Could not load this library: libtorchcodec_core9.dll
```
**Fix Implemented in `scripts/streaming.py`**:
```python
# Cast audio feature to raw binary mode, preventing torchcodec invocation
ds = load_dataset(DATASET_NAME, split="train", streaming=True)
ds = ds.cast_column("audio", datasets.Audio(decode=False))

# Decode raw audio bytes in-memory using libsndfile/soundfile
if "bytes" in audio_data and audio_data["bytes"] is not None:
    import io, soundfile as sf
    waveform, sr = sf.read(io.BytesIO(audio_data["bytes"]))
```
This ensures zero external FFmpeg DLL dependencies and reliable, fast audio ingestion on Windows.

### 3.2 Cache Invalidation & Regeneration
Because V2 draws from completely new row IDs across the 5 TTS generators, previous audio and feature caches were invalidated and purged:
- `Remove-Item -Recurse -Force data\deepfake_cache\audio`
- `Remove-Item -Recurse -Force data\deepfake_cache\features`
- The pipeline now caches clean, verified 16 kHz `.pt` tensors directly mapped to the V2 metadata row IDs.

---

## 4. Repository Knowledge Graph Integration

To maintain architectural visibility and support agentic reasoning, `graphifyy` was installed and configured for the project:
- **Files Analyzed**: 20 files (14 code, 6 documentation).
- **Network Stats**: 125 nodes, 242 edges, 8 communities.
- **Artifacts Generated**:
  - `graphify-out/graph.html`: Standalone interactive D3 visualization of the entire codebase.
  - `graphify-out/GRAPH_REPORT.md`: Comprehensive graph metrics report detailing god nodes and community cohesion.
  - `graphify-out/graph.json`: Full GraphRAG JSON schema for programmatic architecture queries.
  - `.agents/rules/graphify.md` & `.agents/workflows/graphify.md`: Automated rule hooks for keeping the graph synchronized after edits.

---

## 5. Experiment Protocol for V2 Benchmark

The standard experimental evaluation follows the established protocol:
1. **Test Set**: 500 samples (250 bonafide, 250 spoof) sampled with balanced class and generator constraints.
2. **Zero-Shot Evaluation**: Evaluating the pretrained English SLS head (`sukhdeveyash/XLS-R-SLS-Deepfake-Detection`) directly on Hindi speech without weight updates.
3. **Few-Shot Adaptation ($N=10$, $N=50$)**:
   - Fine-tuning only the SLS classification head with frozen XLS-R backbone.
   - Run across 3 deterministic seeds (`1000`, `1001`, `1002`).
   - Strict disjointness between test samples and few-shot training candidates.
   - Nested supervision ($N=10$ candidate pool is a strict subset of $N=50$).

The execution command:
```powershell
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py `
  --parquet-path data\sea_spoof_en_hi_metadata_v2.parquet `
  --device cuda `
  --test-per-class 250
```

---

## 6. Summary of Uncommitted / Recently Added Assets

The following files and updates represent the latest progress:
1. `docs/DATASET_V2_AUDIT_AND_PIPELINE_REPORT.md` (this report)
2. `data/sea_spoof_en_hi_metadata_v2.parquet` (balanced 800-sample dataset)
3. `data/hindi_full_pool.parquet` (16,382 discovered Hindi rows)
4. `scripts/streaming.py` (Windows soundfile streaming fix)
5. `scripts/run_deepfake_detection.py` (V2 parquet default & diagnostics)
6. `scripts/generate_corrected_hindi_subset.py` (automated scanning and stratified sampling tool)
7. `graphify-out/` (complete codebase knowledge graph)
