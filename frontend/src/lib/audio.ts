// Browser-side audio handling: decode any format the browser understands,
// resample to the model's 16 kHz mono, render a spectrogram, encode WAV.

export const SAMPLE_RATE = 16000
export const WINDOW_SAMPLES = 64600 // what the model hears: ~4.04 s
export const WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
export const MAX_SECONDS = 120

export interface Clip {
  name: string
  samples: Float32Array // 16 kHz mono
  duration: number // seconds
  source: 'file' | 'microphone'
}

export async function decodeToClip(blob: Blob, name: string, source: Clip['source']): Promise<Clip> {
  const bytes = await blob.arrayBuffer()
  const ctx = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await ctx.decodeAudioData(bytes)
  } catch {
    throw new Error(`Your browser couldn't decode “${name}”. Try a WAV, MP3, FLAC or OGG file.`)
  } finally {
    void ctx.close()
  }
  const seconds = Math.min(decoded.duration, MAX_SECONDS)
  if (seconds < 0.25) throw new Error('That clip is shorter than a quarter of a second. Use at least one second of speech.')

  const frames = Math.ceil(seconds * SAMPLE_RATE)
  const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE)
  const src = offline.createBufferSource()
  src.buffer = decoded
  src.connect(offline.destination) // the mixdown to one channel happens here
  src.start(0, 0, seconds)
  const rendered = await offline.startRendering()
  return { name, samples: rendered.getChannelData(0).slice(), duration: seconds, source }
}

/** Find the onset time (in seconds) of active speech in the clip to auto-position the analysis window. */
export function findSpeechOnset(samples: Float32Array, rate = SAMPLE_RATE): number {
  if (samples.length <= WINDOW_SAMPLES) return 0

  // 25ms frames with 10ms hop
  const frameLen = Math.round(0.025 * rate)
  const hopLen = Math.round(0.010 * rate)
  const numFrames = Math.floor((samples.length - frameLen) / hopLen)
  if (numFrames <= 0) return 0

  let maxEnergy = 0
  const energies = new Float32Array(numFrames)
  for (let i = 0; i < numFrames; i++) {
    let sumSq = 0
    const offset = i * hopLen
    for (let j = 0; j < frameLen; j++) {
      const val = samples[offset + j]
      sumSq += val * val
    }
    const e = sumSq / frameLen
    energies[i] = e
    if (e > maxEnergy) maxEnergy = e
  }

  if (maxEnergy < 1e-5) return 0

  // Speech threshold: 1% of peak energy (-20 dB)
  const thresh = Math.max(1e-5, maxEnergy * 0.01)
  let onsetFrame = 0
  for (let i = 0; i < numFrames; i++) {
    if (energies[i] >= thresh) {
      onsetFrame = i
      break
    }
  }

  // 150ms pre-onset acoustic buffer so plosives/breathing are preserved
  const onsetSec = Math.max(0, (onsetFrame * hopLen) / rate - 0.15)
  const maxStart = Math.max(0, (samples.length - WINDOW_SAMPLES) / rate)
  return Math.min(onsetSec, maxStart)
}

/** Slice the 4.04 s analysis window starting at `start` seconds. */
export function windowSlice(clip: Clip, start: number): Float32Array {
  const from = Math.max(0, Math.round(start * SAMPLE_RATE))
  return clip.samples.subarray(from, Math.min(clip.samples.length, from + WINDOW_SAMPLES))
}

export function encodeWav(samples: Float32Array, rate = SAMPLE_RATE): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const v = new DataView(buffer)
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i))
  }
  str(0, 'RIFF')
  v.setUint32(4, 36 + samples.length * 2, true)
  str(8, 'WAVE')
  str(12, 'fmt ')
  v.setUint32(16, 16, true)
  v.setUint16(20, 1, true)
  v.setUint16(22, 1, true)
  v.setUint32(24, rate, true)
  v.setUint32(28, rate * 2, true)
  v.setUint16(32, 2, true)
  v.setUint16(34, 16, true)
  str(36, 'data')
  v.setUint32(40, samples.length * 2, true)
  for (let i = 0, o = 44; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

// ---------------------------------------------------------------- spectrogram

const FFT = 512
const HOP = 160 // 10 ms
const BINS = FFT / 2

function fftInPlace(re: Float32Array, im: Float32Array) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k
        const b = a + len / 2
        const xr = re[b] * cr - im[b] * ci
        const xi = re[b] * ci + im[b] * cr
        re[b] = re[a] - xr
        im[b] = im[a] - xi
        re[a] += xr
        im[a] += xi
        const t = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = t
      }
    }
  }
}

export interface Spectrogram {
  frames: number
  bins: number
  /** dB-normalised magnitudes in [0, 1], frame-major */
  data: Float32Array
  hopSeconds: number
}

export function computeSpectrogram(samples: Float32Array): Spectrogram {
  const frames = Math.max(1, Math.floor((samples.length - FFT) / HOP) + 1)
  const data = new Float32Array(frames * BINS)
  const hann = new Float32Array(FFT)
  for (let i = 0; i < FFT; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FFT - 1))
  const re = new Float32Array(FFT)
  const im = new Float32Array(FFT)
  let max = -Infinity
  for (let f = 0; f < frames; f++) {
    const off = f * HOP
    for (let i = 0; i < FFT; i++) {
      re[i] = (samples[off + i] ?? 0) * hann[i]
      im[i] = 0
    }
    fftInPlace(re, im)
    for (let b = 0; b < BINS; b++) {
      const db = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-10)
      data[f * BINS + b] = db
      if (db > max) max = db
    }
  }
  const floor = max - 80
  for (let i = 0; i < data.length; i++) data[i] = Math.max(0, (data[i] - floor) / 80)
  return { frames, bins: BINS, data, hopSeconds: HOP / SAMPLE_RATE }
}

/** Sienna on paper: silence is white, energy deepens toward sienna, like ink on kraft paper. */
export const SIENNA_RAMP = ['#ffffff', '#fdf5ef', '#fbe9dd', '#f6d6c3', '#e9b89e', '#d3937a', '#b46f57', '#8e4c37', '#6e3523', '#5d2a1a', '#3f1c11']

function buildLut(): Uint8ClampedArray {
  const stops = SIENNA_RAMP.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)])
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1)
    const k = Math.min(stops.length - 2, Math.floor(t))
    const u = t - k
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * u
  }
  return lut
}
const LUT = buildLut()

/** Paint the spectrogram into an offscreen canvas, one pixel per frame, rows on a warped scale so speech fills the height. */
export function paintSpectrogram(spec: Spectrogram, height = 256): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = spec.frames
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(spec.frames, height)
  for (let y = 0; y < height; y++) {
    const frac = 1 - y / (height - 1)
    const bin = Math.min(spec.bins - 1, Math.floor(Math.pow(frac, 1.8) * spec.bins))
    for (let x = 0; x < spec.frames; x++) {
      const v = spec.data[x * spec.bins + bin]
      const idx = Math.min(255, Math.floor(Math.pow(v, 1.6) * 255)) * 3
      const p = (y * spec.frames + x) * 4
      img.data[p] = LUT[idx]
      img.data[p + 1] = LUT[idx + 1]
      img.data[p + 2] = LUT[idx + 2]
      img.data[p + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  return canvas
}

/** Peak envelope for the waveform strip: `buckets` pairs of [min, max]. */
export function peaks(samples: Float32Array, buckets: number): Float32Array {
  const out = new Float32Array(buckets * 2)
  const per = samples.length / buckets
  for (let b = 0; b < buckets; b++) {
    let lo = 0
    let hi = 0
    const end = Math.min(samples.length, Math.floor((b + 1) * per))
    for (let i = Math.floor(b * per); i < end; i++) {
      const s = samples[i]
      if (s < lo) lo = s
      if (s > hi) hi = s
    }
    out[b * 2] = lo
    out[b * 2 + 1] = hi
  }
  return out
}

export function formatSeconds(s: number): string {
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}
