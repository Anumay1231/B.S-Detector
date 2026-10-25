import { useEffect, useRef, useState } from 'react'
import { SAMPLE_RATE, WINDOW_SECONDS, computeSpectrogram, decodeToClip, paintSpectrogram, type Clip } from './audio'

export const RECORD_LIMIT = 15

export type ClipPhase = 'idle' | 'decoding' | 'recording' | 'ready'

/** Feeds the live level (0..1) of whatever is playing or recording to the page. */
export type LevelSink = (analyser: AnalyserNode) => () => void

let sharedCtx: AudioContext | null = null
const audioCtx = () => (sharedCtx ??= new AudioContext())

/** One audio slot: load from a file or the microphone, show it, play the analysis window. */
export function useClip(onLevel: LevelSink) {
  const [phase, setPhase] = useState<ClipPhase>('idle')
  const [clip, setClip] = useState<Clip | null>(null)
  const [image, setImage] = useState<HTMLCanvasElement | null>(null)
  const [windowStart, setWindowStart] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [live, setLive] = useState<AnalyserNode | null>(null)
  const [recordSeconds, setRecordSeconds] = useState(0)

  const playheadRef = useRef<number | null>(null)
  const playback = useRef<{ src: AudioBufferSourceNode; stop: () => void } | null>(null)
  const recorder = useRef<{ stop: () => void } | null>(null)

  const stopPlayback = () => {
    playback.current?.stop()
    playback.current = null
  }

  const load = async (blob: Blob, name: string, source: Clip['source']) => {
    stopPlayback()
    setError(null)
    setPhase('decoding')
    try {
      const c = await decodeToClip(blob, name, source)
      // give the browser a frame to paint the decoding state before the FFT
      await new Promise((r) => requestAnimationFrame(() => r(null)))
      setImage(paintSpectrogram(computeSpectrogram(c.samples)))
      setClip(c)
      setWindowStart(0)
      setPhase('ready')
    } catch (e) {
      setError((e as Error).message)
      setPhase(clip ? 'ready' : 'idle')
    }
  }

  const loadFiles = (files: FileList | null) => {
    const f = files?.[0]
    if (f) void load(f, f.name, 'file')
  }

  const startRecording = async () => {
    stopPlayback()
    setError(null)
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
    } catch {
      setError('Microphone access was blocked. Allow it in your browser’s site settings, or choose a file instead.')
      return
    }
    const ac = audioCtx()
    await ac.resume()
    const analyser = ac.createAnalyser()
    analyser.fftSize = 1024
    analyser.smoothingTimeConstant = 0.2
    const srcNode = ac.createMediaStreamSource(stream)
    srcNode.connect(analyser)
    const stopMeter = onLevel(analyser)

    const rec = new MediaRecorder(stream)
    const chunks: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    const started = performance.now()
    const stop = () => {
      if (rec.state !== 'inactive') rec.stop()
    }
    const timer = window.setInterval(() => {
      const s = (performance.now() - started) / 1000
      setRecordSeconds(s)
      if (s >= RECORD_LIMIT) stop()
    }, 100)
    rec.onstop = () => {
      window.clearInterval(timer)
      stopMeter()
      srcNode.disconnect()
      stream.getTracks().forEach((t) => t.stop())
      setLive(null)
      recorder.current = null
      void load(new Blob(chunks, { type: rec.mimeType }), 'Microphone recording', 'microphone')
    }
    recorder.current = { stop }
    rec.start()
    setRecordSeconds(0)
    setLive(analyser)
    setPhase('recording')
  }

  const stopRecording = () => recorder.current?.stop()

  /** Play the 4.04 s window (or the whole clip when `whole` is set). */
  const togglePlay = async (whole = false) => {
    if (!clip) return
    if (playback.current) {
      stopPlayback()
      return
    }
    const ac = audioCtx()
    await ac.resume()
    const buffer = ac.createBuffer(1, clip.samples.length, SAMPLE_RATE)
    buffer.copyToChannel(clip.samples as Float32Array<ArrayBuffer>, 0)
    const src = ac.createBufferSource()
    src.buffer = buffer
    const analyser = ac.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser).connect(ac.destination)
    const from = whole ? 0 : windowStart
    const len = whole ? clip.duration : Math.min(WINDOW_SECONDS, clip.duration)
    const t0 = ac.currentTime
    src.start(0, from, len)
    const stopMeter = onLevel(analyser)
    let raf = 0
    const follow = () => {
      playheadRef.current = from + (ac.currentTime - t0)
      raf = requestAnimationFrame(follow)
    }
    follow()
    const stop = () => {
      cancelAnimationFrame(raf)
      stopMeter()
      try {
        src.stop()
      } catch {
        /* already stopped */
      }
      playheadRef.current = null
      setPlaying(false)
    }
    src.onended = () => {
      if (playback.current?.src === src) playback.current = null
      stop()
    }
    playback.current = { src, stop }
    setPlaying(true)
  }

  useEffect(
    () => () => {
      stopPlayback()
      recorder.current?.stop()
    },
    [],
  )

  return {
    phase,
    clip,
    image,
    windowStart,
    setWindowStart,
    error,
    setError,
    playing,
    live,
    recordSeconds,
    playheadRef,
    loadFiles,
    startRecording,
    stopRecording,
    togglePlay,
    stopPlayback,
    busy: phase === 'decoding' || phase === 'recording',
  }
}

export type ClipSlot = ReturnType<typeof useClip>
