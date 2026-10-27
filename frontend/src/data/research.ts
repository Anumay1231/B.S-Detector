// Stage 1 results from outputs/results_full_10_50_500.json (the current benchmark):
// a held-out, balanced test set of 200 real and 200 generated Hindi clips. Few-shot rows
// are the mean of 3 seeds, ± one (population) standard deviation as in the team's README.
// Fakes caught and real voices passed are the mean of the per-seed confusion matrices.
// The unfreezing ablation below comes from earlier runs on a 439-clip split
// (outputs/results_unfreeze_*.json), matching the team's figure 3.

export interface Regime {
  id: string
  label: string
  shots: string
  /** clips per class used to adapt the head */
  perClass: number
  accuracy: number
  accuracyStd?: number
  eer: number
  eerStd?: number
  precision: number
  recall: number
  f1: number
  genuinePassed: number
  fakesCaught: number
}

export const regimes: Regime[] = [
  { id: 'zero', label: 'English-trained, no Hindi', shots: '0 Hindi clips', perClass: 0, accuracy: 0.6175, eer: 0.3225, precision: 0.567, recall: 0.995, f1: 0.7223, genuinePassed: 199, fakesCaught: 48 },
  { id: 'n10', label: 'Adapted on 10 per class', shots: '20 Hindi clips', perClass: 10, accuracy: 0.6433, accuracyStd: 0.0504, eer: 0.2764, eerStd: 0.0227, precision: 0.5872, recall: 0.99, f1: 0.7363, genuinePassed: 198, fakesCaught: 59 },
  { id: 'n50', label: 'Adapted on 50 per class', shots: '100 Hindi clips', perClass: 50, accuracy: 0.7583, accuracyStd: 0.1049, eer: 0.1818, eerStd: 0.0118, precision: 0.7169, recall: 0.9317, f1: 0.8016, genuinePassed: 186, fakesCaught: 117 },
  { id: 'n500', label: 'Adapted on 500 per class', shots: '1,000 Hindi clips', perClass: 500, accuracy: 0.9683, accuracyStd: 0.0082, eer: 0.0266, eerStd: 0.0052, precision: 0.9578, recall: 0.98, f1: 0.9687, genuinePassed: 196, fakesCaught: 191 },
]

/** The best single N=500 run (seed 2), as quoted by the team. */
export const bestRun = { accuracy: 0.98, eer: 0.0212, fakesMissed: 6, falseAlarms: 2 }

export interface Ablation {
  id: string
  label: string
  trainable: string
  layers: number
  accuracy: number
  accuracyStd: number
  eer: number
  eerStd: number
  /** one seed's confusion matrix: [[spoof→spoof, spoof→genuine], [genuine→spoof, genuine→genuine]] */
  matrix?: [[number, number], [number, number]]
}

export const ablations: Ablation[] = [
  { id: 'frozen', label: 'Backbone frozen', trainable: '23.4M', layers: 0, accuracy: 0.7967, accuracyStd: 0.0047, eer: 0.197, eerStd: 0.0057 },
  { id: 'top2', label: 'Top 2 layers unfrozen', trainable: '50.8M', layers: 2, accuracy: 0.7646, accuracyStd: 0.0047, eer: 0.2414, eerStd: 0.0158, matrix: [[90, 99], [2, 248]] },
  { id: 'top4', label: 'Top 4 layers unfrozen', trainable: '78.8M', layers: 4, accuracy: 0.6712, accuracyStd: 0.0583, eer: 0.2531, eerStd: 0.0191, matrix: [[27, 162], [0, 250]] },
]

export interface Slice {
  label: string
  detail: string
  count: number
  kind: 'genuine' | 'synthetic'
}

export const benchmark: Slice[] = [
  { label: 'Real Hindi voices', detail: 'Studio (IndicTTS) and crowdsourced (Common Voice)', count: 200, kind: 'genuine' },
  { label: 'Edge-TTS', detail: 'Microsoft neural cloud voice', count: 40, kind: 'synthetic' },
  { label: 'VITS-MMS', detail: 'Meta end-to-end VITS', count: 40, kind: 'synthetic' },
  { label: 'Indic-TTS', detail: 'IIT Madras synthesis', count: 40, kind: 'synthetic' },
  { label: 'FastSpeech', detail: 'Non-autoregressive transformer', count: 40, kind: 'synthetic' },
  { label: 'XTTS-v2', detail: 'Zero-shot voice cloning', count: 40, kind: 'synthetic' },
]

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`

// Stage 2, speaker verification: docs/calibration.md, Phases 8B and 8C.
// ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) used as published, scored with
// cosine similarity. These are a VoxCeleb1 English baseline, not Hindi results.

/** Phase 8B: speaker-disjoint split. Its EER threshold is the one the app ships. */
export const speakerBaseline = {
  trials: 28280,
  speakers: 40,
  utterances: 4715,
  calibration: { genuine: 10968, impostor: 6192 },
  evaluation: { genuine: 7892, impostor: 3228 },
  eer: 0.008721,
  threshold: 0.252784,
  rocAuc: 0.998634,
  /** measured on held-out speakers the threshold never saw */
  heldOutFar: 0.011462,
  heldOutFrr: 0.004815,
  genuineMean: 0.590961,
  impostorMean: 0.023593,
  /** calibration split, the scores the threshold was chosen on */
  calibrationGenuineMean: 0.593241,
  calibrationImpostorMean: 0.020799,
  calibrationImpostorMax: 0.407291,
}

/** Phase 8C: the full official VoxCeleb1 trial list, for comparison with published results. */
export const speakerProtocol = {
  trials: 37720,
  eer: 0.010392,
  threshold: 0.272217,
  rocAuc: 0.998945,
  /** the shipped 8B threshold applied to the full list */
  farAtShipped: 0.01474,
  frrAtShipped: 0.007105,
  /** SpeechBrain's published figure, reported on the cleaned trial list */
  publishedEer: 0.008,
}

// Phase 9: docs/speaker-verification-hindi-english.md. ECAPA-TDNN fused with
// WeSpeaker ResNet34 on Hindi and Indian-English speakers: segment-averaged
// embeddings, a duration-aware threshold and an unsure band, fitted on Common
// Voice + SVQ and tested on held-out SVQ speakers (2-fold).
export const speakerPhase9 = {
  /** accuracy on the held-out SVQ pairs it answers with >= 8 s of speech, over hi/en and clean/same-room */
  accuracyReliable: '96.9 to 100%',
  /** same-gender impostors the VoxCeleb cut-off (0.2528) accepted on held-out SVQ */
  oldImpostorsAccepted: '42 to 70%',
}
