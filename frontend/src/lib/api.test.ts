import { describe, expect, it } from 'vitest'
import { speakerMeter } from './api'

describe('speakerMeter', () => {
  it('puts a score on the cut-off in the middle', () => {
    expect(speakerMeter({ score: 2.8, threshold: 2.8, margin: 0.4, system: 'fused' })).toBe(0.5)
    expect(speakerMeter({ score: 0.25, threshold: 0.25, system: 'ecapa' })).toBe(0.5)
  })

  it('spreads fused scores over 3 margins, at least 2.5 each way', () => {
    // margin 1 -> span 3: a score 3 above the cut-off reaches the right end
    expect(speakerMeter({ score: 5, threshold: 2, margin: 1, system: 'fused' })).toBeCloseTo(1)
    // margin 0.4 -> 3 x 0.4 = 1.2 is under the floor, so the span is 2.5
    expect(speakerMeter({ score: -0.5, threshold: 2, margin: 0.4, system: 'fused' })).toBeCloseTo(0)
    expect(speakerMeter({ score: 2, threshold: 2, system: 'fused' })).toBe(0.5)
  })

  it('spreads plain ECAPA cosines over 0.5 each way', () => {
    expect(speakerMeter({ score: 0.75, threshold: 0.25, system: 'ecapa' })).toBeCloseTo(1)
    expect(speakerMeter({ score: 0, threshold: 0.25 })).toBeCloseTo(0.25)
  })

  it('does not clamp: callers decide what to do past the ends', () => {
    expect(speakerMeter({ score: 10, threshold: 2, margin: 0.4, system: 'fused' })).toBeGreaterThan(1)
  })
})
