// Results from outputs/results_balanced_200.json and the unfreezing ablations
// (outputs/results_unfreeze_2layers_balanced.json, outputs/results_unfreeze_l4.json).
// Balanced V2 test set: 200 genuine + 200 synthetic Hindi clips. Few-shot rows are
// the mean of 3 seeds. "Caught" counts are derived from accuracy and genuine-class
// recall: genuine passed = recall × 200, fakes caught = accuracy × 400 − genuine passed.

export interface Regime {
  id: string
  label: string
  shots: string
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

const derive = (accuracy: number, recall: number) => {
  const genuinePassed = Math.round(recall * 200)
  return { genuinePassed, fakesCaught: Math.round(accuracy * 400) - genuinePassed }
}

export const regimes: Regime[] = [
  { id: 'zero', label: 'English-trained, no Hindi', shots: '0 Hindi clips', accuracy: 0.6, eer: 0.289, precision: 0.5568, recall: 0.98, f1: 0.7101, ...derive(0.6, 0.98) },
  {
    id: 'n10',
    label: 'Adapted on 10 per class',
    shots: '20 Hindi clips',
    accuracy: 0.6583,
    accuracyStd: 0.0587,
    eer: 0.2688,
    eerStd: 0.0183,
    precision: 0.6081,
    recall: 0.945,
    f1: 0.7361,
    ...derive(0.6583, 0.945),
  },
  {
    id: 'n50',
    label: 'Adapted on 50 per class',
    shots: '100 Hindi clips',
    accuracy: 0.7967,
    accuracyStd: 0.0047,
    eer: 0.197,
    eerStd: 0.0057,
    precision: 0.7882,
    recall: 0.8183,
    f1: 0.8005,
    ...derive(0.7967, 0.8183),
  },
]

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
  { id: 'frozen', label: 'Backbone frozen', trainable: '22.8M', layers: 0, accuracy: 0.7967, accuracyStd: 0.0047, eer: 0.197, eerStd: 0.0057 },
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
  { label: 'IndicTTS recordings', detail: 'Studio speech', count: 120, kind: 'genuine' },
  { label: 'Mozilla Common Voice', detail: 'Crowdsourced, real-world mics', count: 80, kind: 'genuine' },
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
