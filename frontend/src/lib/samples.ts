import type { Analysis, TwoStageResult } from './api'

// Bundled demo clips live in public/samples (see CREDITS.md there).

export interface Sample {
  id: string
  file: string
  title: string
  language: string
  kind: 'real' | 'generated'
  source: string
  license: string
  transcript: string
  durationSec: number
}

export interface SamplePair {
  id: string
  title: string
  reference: string
  test: string
  expected: 'same' | 'different'
}

export interface Manifest {
  samples: Sample[]
  pairs: SamplePair[]
  /** API responses recorded when the samples were made */
  results: {
    recordedAt?: string
    analyze: Record<string, Analysis>
    verify: Record<string, TwoStageResult>
  }
}

export async function loadManifest(): Promise<Manifest | null> {
  try {
    const res = await fetch('/samples/manifest.json')
    if (!res.ok) return null
    return (await res.json()) as Manifest
  } catch {
    return null
  }
}

export async function sampleBlob(file: string): Promise<Blob> {
  const res = await fetch(`/samples/${file}`)
  if (!res.ok) throw new Error(`The sample “${file}” is missing from this build.`)
  return res.blob()
}

/** Short labels for the sample buttons. */
export const SAMPLE_LABELS: Record<string, string> = {
  'real-hindi': 'Real Hindi voice',
  'fake-hindi-xtts': 'XTTS-v2 fake, same sentence',
  'same-speaker': 'Same person, two clips',
  'different-speaker': 'Two different people',
  'hindi-real-vs-xtts': 'Real voice vs its XTTS-v2 fake',
}
