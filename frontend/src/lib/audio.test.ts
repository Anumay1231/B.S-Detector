import { describe, expect, it } from 'vitest'
import { SAMPLE_RATE, WINDOW_SAMPLES, encodeWav, formatSeconds, windowSlice, type Clip } from './audio'

const clip = (seconds: number): Clip => {
  const samples = new Float32Array(Math.round(seconds * SAMPLE_RATE)).map((_, i) => i / SAMPLE_RATE)
  return { name: 'test', samples, duration: seconds, source: 'file' }
}

describe('encodeWav', () => {
  it('writes a 16-bit mono PCM header for the given rate', async () => {
    const v = new DataView(await encodeWav(new Float32Array(100)).arrayBuffer())
    const text = (o: number) => String.fromCharCode(...new Uint8Array(v.buffer, o, 4))
    expect(text(0)).toBe('RIFF')
    expect(text(8)).toBe('WAVE')
    expect(text(12)).toBe('fmt ')
    expect(text(36)).toBe('data')
    expect(v.getUint32(4, true)).toBe(36 + 200)
    expect(v.getUint16(20, true)).toBe(1) // PCM
    expect(v.getUint16(22, true)).toBe(1) // mono
    expect(v.getUint32(24, true)).toBe(SAMPLE_RATE)
    expect(v.getUint32(28, true)).toBe(SAMPLE_RATE * 2)
    expect(v.getUint16(34, true)).toBe(16)
    expect(v.getUint32(40, true)).toBe(200)
    expect(v.byteLength).toBe(244)
  })

  it('uses the rate it is given', async () => {
    const v = new DataView(await encodeWav(new Float32Array(10), 8000).arrayBuffer())
    expect(v.getUint32(24, true)).toBe(8000)
  })

  it('scales samples to int16 and clips anything past full scale', async () => {
    const v = new DataView(await encodeWav(new Float32Array([0, 1, -1, 0.5, 2, -3])).arrayBuffer())
    const at = (i: number) => v.getInt16(44 + 2 * i, true)
    expect([at(0), at(1), at(2), at(3), at(4), at(5)]).toEqual([0, 32767, -32768, 16383, 32767, -32768])
  })
})

describe('windowSlice', () => {
  it('takes 4.04 s from the start', () => {
    const w = windowSlice(clip(10), 0)
    expect(w.length).toBe(WINDOW_SAMPLES)
    expect(w[0]).toBe(0)
  })

  it('starts at the given second', () => {
    const w = windowSlice(clip(10), 2)
    expect(w.length).toBe(WINDOW_SAMPLES)
    expect(w[0]).toBeCloseTo(2, 5)
  })

  it('stops at the end of a short clip', () => {
    expect(windowSlice(clip(3), 0).length).toBe(3 * SAMPLE_RATE)
    expect(windowSlice(clip(5), 4).length).toBe(SAMPLE_RATE)
  })

  it('treats a negative start as 0', () => {
    expect(windowSlice(clip(10), -1)[0]).toBe(0)
  })
})

describe('formatSeconds', () => {
  it('shows two decimals under 10 s, one under a minute', () => {
    expect(formatSeconds(0)).toBe('0.00 s')
    expect(formatSeconds(4.0375)).toBe('4.04 s')
    expect(formatSeconds(17.52)).toBe('17.5 s')
  })

  it('switches to m:ss from a minute', () => {
    expect(formatSeconds(60)).toBe('1:00')
    expect(formatSeconds(125.9)).toBe('2:05')
  })
})
