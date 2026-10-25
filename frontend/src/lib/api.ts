export type ModelState = 'connecting' | 'loading' | 'ready' | 'error' | 'unavailable' | 'offline'

export interface Status {
  state: ModelState
  message: string
  device: string | null
  checkpoint: string
  windowSeconds: number
}

export interface Analysis {
  verdict: 'SPOOF' | 'BONAFIDE'
  probSpoof: number
  probBonafide: number
  layerWeights: number[]
  latencyMs: number
  durationSec: number
  device: string
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

export async function analyze(wav: Blob): Promise<Analysis> {
  let res: Response
  try {
    res = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav })
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
  return (await res.json()) as Analysis
}
