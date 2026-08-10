"""
calibration.py
---------------

Intended purpose (baseline architecture only):
    Threshold calibration and evaluation utilities for the ECAPA-TDNN
    cosine-similarity baseline. Scope is strictly limited to:
        - genuine/impostor trial score handling
        - threshold calibration (selecting decision threshold(s) from
          genuine/impostor score distributions)
        - False Acceptance Rate (FAR)
        - False Rejection Rate (FRR)
        - Equal Error Rate (EER)
        - ROC curve / ROC-AUC computation
        - calibrated threshold(s) used to produce MATCH / NON_MATCH /
          UNCERTAIN decisions

    Fuzzy logic is explicitly OUT OF SCOPE for this module. It is not part
    of the baseline calibration pipeline and must not be implemented or
    referenced here. Fuzzy logic will live in a separate, optional,
    experimental module introduced later, only after the ECAPA-TDNN
    baseline has been fully evaluated.

Status: placeholder only. No implementation yet.
"""

# TODO: implement genuine/impostor trial score handling here.
# TODO: implement threshold calibration here.
# TODO: implement FAR / FRR computation here.
# TODO: implement EER computation here.
# TODO: implement ROC curve / ROC-AUC computation here.
# TODO: implement calibrated threshold selection/storage here.
#
# Do NOT implement fuzzy logic in this file.
