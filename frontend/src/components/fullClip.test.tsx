import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { ResultCard } from './ResultCard'
import { VoiceInput } from './VoiceInput'
import type { ClipSlot } from '../lib/useClip'
import type { Run } from '../lib/runs'

const run: Run = {
  id: 1, at: new Date(), mode: 'detect', clipName: 'recording.wav', clipDuration: 30, windowStart: 0,
  analysis: {
    verdict: 'BONAFIDE', probSpoof: 0.475, probBonafide: 0.525, layerWeights: [], latencyMs: 1,
    durationSec: 30, device: 'cpu', head: 'hindi-robust', numWindows: 27, aggregation: 'sliding_window_top2',
    peakSpoofWindow: { index: 26, start_sec: 25.962, end_sec: 30, prob_spoof: 0.9, prob_bonafide: 0.1, verdict: 'SPOOF' },
  },
}

describe('full recording interface', () => {
  it('reports complete coverage and a suspicious tail despite a passing overall score', () => {
    const html = renderToStaticMarkup(<ResultCard run={run} />)
    expect(html).toContain('27')
    expect(html).toContain('Scanned the entire 30.0 s recording')
    expect(html).toContain('Suspicious segment found:')
    expect(html).toContain('26.0 s to 30.0 s')
    expect(html).toContain('overall score is below the flag threshold')
    expect(html).not.toContain('a pause before the speech')
  })

  it('keeps playback but removes the draggable window and manual instructions', () => {
    const slot = {
      clip: { name: 'recording.wav', duration: 30, samples: new Float32Array(1), source: 'file' },
      phase: 'ready', busy: false, image: null, windowStart: 0,
      playheadRef: { current: null }, playing: false, live: null, error: null,
    } as ClipSlot
    const html = renderToStaticMarkup(<VoiceInput slot={slot} title="Voice" helper="" placeholder="" locked={false} />)
    expect(html).not.toContain('role="slider"')
    expect(html).not.toContain('Drag the bracket')
    expect(html).toContain('Play the clip')
    expect(html).toContain('scans the entire recording automatically')
  })
})
