export type ModelState = 'connecting' | 'loading' | 'ready' | 'error' | 'unavailable' | 'offline'

export interface VerifierStatus {
  state: ModelState
  message: string
  device: string | null
  model: string
  threshold: number
  isCalibrated: boolean
}

export interface Status {
  state: ModelState
  message: string
  device: string | null
  checkpoint: string
  windowSeconds: number
  /** which SLS head is loaded: the English checkpoint or the team's Hindi-adapted one */
  head?: 'english' | 'hindi-n500' | 'hindi-robust' | 'hindi-phase12' | 'hindi-channel-robust' | 'hindi-indicsynth' | 'hindi-custom'
  headNote?: string | null
  verifier?: VerifierStatus
}

export interface Analysis {
  verdict: 'SPOOF' | 'BONAFIDE'
  probSpoof: number
  probBonafide: number
  layerWeights: number[]
  latencyMs: number
  durationSec: number
  device: string
  head?: 'english' | 'hindi-n500' | 'hindi-robust' | 'hindi-phase12' | 'hindi-channel-robust' | 'hindi-indicsynth' | 'hindi-custom'
}

export interface Verification {
  /** UNCERTAIN: the score sits too close to the cut-off to call */
  outcome: 'MATCH' | 'NON_MATCH' | 'UNCERTAIN'
  /** 'fused': ECAPA + ResNet34 z-scores averaged; 'ecapa': the plain ECAPA cosine, about -1..1 */
  score: number
  threshold: number
  /** half-width of the UNCERTAIN band around the threshold */
  margin?: number
  system?: 'fused' | 'ecapa'
  cosines?: { ecapa: number; wespeaker?: number }
  isCalibrated: boolean
  latencyMs: number
  /** seconds of speech left in each clip after silence is trimmed */
  referenceSeconds: number
  testSeconds: number
  /** both clips had enough speech for the calibrated accuracy to apply */
  reliable?: boolean
  reliableSeconds?: number
}

/** Speaker score on a 0..1 meter with the cut-off in the middle. */
export function speakerMeter(v: Pick<Verification, 'score' | 'threshold' | 'margin' | 'system'>): number {
  const span = v.system === 'fused' ? Math.max(3 * (v.margin ?? 0), 2.5) : 0.5
  return 0.5 + (v.score - v.threshold) / (2 * span)
}

export interface TwoStageResult {
  detection: Analysis
  /** stage 1 on the known voice's first 4.04 s, so a cloned reference can't vouch for a clone */
  referenceDetection?: Analysis
  /** true when stage 1 called a voice synthetic, so stage 2 never ran */
  interlocked: boolean
  /** which clip tripped the interlock: the voice to check or the known voice */
  interlockReason?: 'window' | 'reference' | null
  verification: Verification | null
}

export async function fetchStatus(): Promise<Status> {
  try {
    const res = await fetch('/api/status', { cache: 'no-store' })
    if (!res.ok) throw new Error(String(res.status))
    return (await res.json()) as Status
  } catch {
    return {
      state: 'offline',
      message: "The local server isn't running. Start it with `python run.py`.",
      device: null,
      checkpoint: 'v1',
      windowSeconds: 4.0375,
    }
  }
}

async function post<T>(url: string, init: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, { method: 'POST', ...init })
  } catch {
    throw new Error("Couldn't reach the local server. Check that `python run.py` is still running.")
  }
  if (!res.ok) {
    let detail = `The server returned ${res.status}.`
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail
    } catch {
      /* keep the generic message */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export const analyze = (wav: Blob) => post<Analysis>('/api/analyze', { headers: { 'Content-Type': 'audio/wav' }, body: wav })

export function verifySpeaker(reference: Blob, test: Blob, windowStart: number) {
  const form = new FormData()
  form.append('reference', reference, 'reference.wav')
  form.append('test', test, 'test.wav')
  form.append('windowStart', String(windowStart))
  return post<TwoStageResult>('/api/verify', { body: form })
}
