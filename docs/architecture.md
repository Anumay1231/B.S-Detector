# Architecture

Status: placeholder. This document will describe the detailed design of the
speaker verification module as it is implemented.

## Pipeline overview

```
Reference Audio
       ↓
Audio Preprocessing
       ↓
ECAPA-TDNN
       ↓
Reference Embedding

Test Audio
       ↓
Audio Preprocessing
       ↓
ECAPA-TDNN
       ↓
Test Embedding

Reference + Test Embeddings
       ↓
Cosine Similarity
       ↓
Threshold Calibration
       ↓
MATCH / NON_MATCH / UNCERTAIN
```

## Integration with B.S. Detector

This module is invoked conditionally by the larger B.S. Detector pipeline,
after the deepfake detection stage, as an identity-consistency check on
audio that has already passed deepfake screening.

## Sections to be filled in as implementation proceeds

- Audio preprocessing details (`audio.py`)
- ECAPA-TDNN encoder details (`encoder.py`)
- Similarity scoring details (`similarity.py`)
- Threshold calibration methodology (`calibration.py`): genuine/impostor
  trial scores, threshold calibration, FAR, FRR, EER, ROC/ROC-AUC
- Evaluation methodology (`evaluation.py`)
- B.S. Detector pipeline integration contract

## Out of scope for the baseline

Fuzzy logic decision boundaries are explicitly out of scope for the
ECAPA-TDNN baseline and for `calibration.py`. Fuzzy logic will be considered
only as a separate, optional, experimental module introduced later, after
the baseline has been fully evaluated.
