# Graph Report - BS-Detector  (2026-09-24)

## Corpus Check
- Corpus is ~12,841 words - fits in a single context window. You may not need a graph.

## Summary
- 125 nodes · 242 edges · 8 communities (7 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Acoustic Feature Extraction
- SLS Model Fine-Tuning & Evaluation
- EER & Detection Metrics
- Experiment Pipeline & Splits
- Dataset Auditing & Subset Generation
- Dataset Streaming & HTTP APIs
- Feature Cache Management
- Package Initialization

## God Nodes (most connected - your core abstractions)
1. `main()` - 13 edges
2. `_run_inference()` - 10 edges
3. `evaluate_zeroshot()` - 10 edges
4. `train_fewshot()` - 10 edges
5. `extract_and_cache_features()` - 9 edges
6. `SLSHead` - 9 edges
7. `run_fewshot_experiment()` - 8 edges
8. `load_features_batch()` - 8 edges
9. `compute_all_metrics()` - 7 edges
10. `print_metrics()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `main()` --indirect_call--> `f()`  [INFERRED]
  scripts/run_deepfake_detection.py → scripts/metrics.py
- `_run_inference()` --calls--> `load_features_batch()`  [EXTRACTED]
  scripts/experiment.py → scripts/features.py
- `_run_inference()` --calls--> `compute_all_metrics()`  [EXTRACTED]
  scripts/experiment.py → scripts/metrics.py
- `_run_inference()` --uses--> `SLSHead`  [INFERRED]
  scripts/experiment.py → scripts/model.py
- `evaluate_zeroshot()` --calls--> `print_metrics()`  [EXTRACTED]
  scripts/experiment.py → scripts/metrics.py

## Import Cycles
- None detected.

## Communities (8 total, 1 thin omitted)

### Community 0 - "Acoustic Feature Extraction"
Cohesion: 0.12
Nodes (25): logging, pathlib, extract_and_cache_features(), Extract and cache XLS-R hidden states for audio clips. Runs the frozen XLS-R…, Extract XLS-R hidden states for a list of clips and cache to disk. For each…, extract_hidden_states(), load_backbone(), pad_or_truncate() (+17 more)

### Community 1 - "SLS Model Fine-Tuning & Evaluation"
Cohesion: 0.16
Nodes (19): copy, evaluate_zeroshot(), Any, Path, Experiment functions for zero-shot and few-shot deepfake detection. Provides: -…, Fine-tune a fresh SLS head on a few-shot training set, then evaluate.…, Run few-shot fine-tuning across multiple seeds for a given N. For each seed,…, Run inference on a set of IDs and return metrics. Args: sls_head: SLSHead… (+11 more)

### Community 2 - "EER & Detection Metrics"
Cohesion: 0.12
Nodes (18): ndarray, scipy_interpolate, scipy_optimize, compute_all_metrics(), compute_eer(), f(), print_comparison_table(), print_metrics() (+10 more)

### Community 3 - "Experiment Pipeline & Splits"
Cohesion: 0.16
Nodes (16): main(), create_fewshot_pools(), create_test_split(), get_all_needed_ids(), get_labels_for_ids(), load_hindi_metadata(), Any, DataFrame (+8 more)

### Community 4 - "Dataset Auditing & Subset Generation"
Cohesion: 0.18
Nodes (13): argparse, collections, datasets, # IMPORTANT:, collect_hindi_metadata(), main(), print_diagnostics(), DataFrame (+5 more)

### Community 5 - "Dataset Streaming & HTTP APIs"
Cohesion: 0.21
Nodes (9): huggingface_hub, json, numpy, os, pandas, requests, scipy_io_wavfile, Phase 2: Dense scan for more Hindi bonafide rows, then create corrected subset.… (+1 more)

### Community 6 - "Feature Cache Management"
Cohesion: 0.47
Nodes (6): load_cached_features(), load_features_batch(), Path, Tensor, Load cached hidden states for a single clip. Args: row_id: The row_id of the…, Load cached features for multiple clips. Args: row_ids: List of row_id strings.…

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Experiment Pipeline & Splits` to `Acoustic Feature Extraction`, `SLS Model Fine-Tuning & Evaluation`, `EER & Detection Metrics`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `SLSHead` connect `SLS Model Fine-Tuning & Evaluation` to `Acoustic Feature Extraction`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `evaluate_zeroshot()` connect `SLS Model Fine-Tuning & Evaluation` to `Acoustic Feature Extraction`, `EER & Detection Metrics`, `Experiment Pipeline & Splits`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Should `Acoustic Feature Extraction` be split into smaller, more focused modules?**
  _Cohesion score 0.1206896551724138 - nodes in this community are weakly interconnected._
- **Should `EER & Detection Metrics` be split into smaller, more focused modules?**
  _Cohesion score 0.12105263157894737 - nodes in this community are weakly interconnected._