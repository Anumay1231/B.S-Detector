import { useEffect, useRef, useState } from 'react'
import {
  MAX_SECONDS,
  SAMPLE_RATE,
  WINDOW_SECONDS,
  computeSpectrogram,
  decodeToClip,
  encodeWav,
  formatSeconds,
  paintSpectrogram,
  windowSlice,
  type Clip,
} from '../lib/audio'
import { analyze, type Status } from '../lib/api'
import { Scope } from './Scope'
import { Verdict, type Run } from './Verdict'
import { SessionLog } from './SessionLog'
import './Analyzer.css'

const RECORD_LIMIT = 15

type Phase = 'idle' | 'decoding' | 'recording' | 'ready' | 'analyzing'

export function Analyzer({ status }: { status: Status }) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [clip, setClip] = useState<Clip | null>(null)
  const [image, setImage] = useState<HTMLCanvasElement | null>(null)
  const [windowStart, setWindowStart] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [playing, setPlaying] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [live, setLive] = useState<AnalyserNode | null>(null)
  const [recordSeconds, setRecordSeconds] = useState(0)

  const titleRef = useRef<HTMLHeadingElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const playheadRef = useRef<number | null>(null)
  const audioCtx = useRef<AudioContext | null>(null)
  const playback = useRef<{ src: AudioBufferSourceNode; stop: () => void } | null>(null)
  const recorder = useRef<{ stop: () => void } | null>(null)

  const latest = runs[0]
  const ready = status.state === 'ready'

  const ctx = () => (audioCtx.current ??= new AudioContext())

  // The headline breathes with whatever is playing or being recorded.
  const meter = (analyser: AnalyserNode) => {
    const buf = new Float32Array(analyser.fftSize)
    let raf = 0
    let smooth = 0
    const tick = () => {
      analyser.getFloatTimeDomainData(buf)
      let sum = 0
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i]
      const rms = Math.sqrt(sum / buf.length)
      smooth = smooth * 0.75 + Math.min(1, rms * 5) * 0.25
      titleRef.current?.style.setProperty('--level', smooth.toFixed(3))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      titleRef.current?.style.setProperty('--level', '0')
    }
  }

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

  const onFiles = (files: FileList | null) => {
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
    const ac = ctx()
    await ac.resume()
    const analyser = ac.createAnalyser()
    analyser.fftSize = 1024
    analyser.smoothingTimeConstant = 0.2
    const srcNode = ac.createMediaStreamSource(stream)
    srcNode.connect(analyser)
    const stopMeter = meter(analyser)

    const rec = new MediaRecorder(stream)
    const chunks: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    const started = performance.now()
    const timer = window.setInterval(() => {
      const s = (performance.now() - started) / 1000
      setRecordSeconds(s)
      if (s >= RECORD_LIMIT) stop()
    }, 100)
    const stop = () => {
      if (rec.state === 'inactive') return
      rec.stop()
    }
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

  const togglePlay = async () => {
    if (!clip) return
    if (playback.current) {
      stopPlayback()
      return
    }
    const ac = ctx()
    await ac.resume()
    const buffer = ac.createBuffer(1, clip.samples.length, SAMPLE_RATE)
    buffer.copyToChannel(clip.samples as Float32Array<ArrayBuffer>, 0)
    const src = ac.createBufferSource()
    src.buffer = buffer
    const analyser = ac.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser).connect(ac.destination)
    const len = Math.min(WINDOW_SECONDS, clip.duration)
    const t0 = ac.currentTime
    src.start(0, windowStart, len)
    const stopMeter = meter(analyser)
    let raf = 0
    const follow = () => {
      playheadRef.current = windowStart + (ac.currentTime - t0)
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

  const runAnalysis = async () => {
    if (!clip || !ready) return
    stopPlayback()
    setError(null)
    setPhase('analyzing')
    try {
      const analysis = await analyze(encodeWav(windowSlice(clip, windowStart)))
      setRuns((r) => [{ id: Date.now(), clipName: clip.name, clipDuration: clip.duration, windowStart, analysis, at: new Date() }, ...r])
      requestAnimationFrame(() => document.getElementById('verdict')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setPhase('ready')
    }
  }

  useEffect(() => () => {
    stopPlayback()
    recorder.current?.stop()
    void audioCtx.current?.close()
  }, [])

  const busy = phase === 'decoding' || phase === 'analyzing' || phase === 'recording'
  const windowEnd = clip ? Math.min(clip.duration, windowStart + WINDOW_SECONDS) : 0

  let checkLabel = 'Check this clip'
  if (phase === 'analyzing') checkLabel = 'Listening…'
  else if (!ready) checkLabel = status.state === 'loading' || status.state === 'connecting' ? 'Model still loading' : 'Model unavailable'

  return (
    <section className="analyzer page" id="analyze" aria-labelledby="hero-title">
      <header className="hero">
        <h1 className="hero-title" id="hero-title" ref={titleRef}>
          <span>Is this voice</span> <span>real?</span>
        </h1>
        <p className="hero-hindi" lang="hi">
          क्या यह आवाज़ असली है?
        </p>
        <p className="hero-lede">
          Give it a clip of speech. A frozen XLS-R 300M model listens to about four seconds and tells you whether the voice sounds human or
          generated, and which of its 24 layers it relied on to decide.
        </p>
      </header>

      <div
        className={`scope${dragOver ? ' is-dragover' : ''}${clip ? ' has-clip' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          if (!busy) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          if (!busy) onFiles(e.dataTransfer.files)
        }}
      >
        <div className="scope-head">
          <div className="scope-title">
            {phase === 'recording' ? (
              <>
                <span className="rec-dot" aria-hidden="true" />
                <span>Recording</span>
                <span className="scope-meta">
                  {recordSeconds.toFixed(1)} of {RECORD_LIMIT} s
                </span>
              </>
            ) : clip ? (
              <>
                <span className="scope-name" title={clip.name}>
                  {clip.name}
                </span>
                <span className="scope-meta">{formatSeconds(clip.duration)}</span>
              </>
            ) : (
              <span className="scope-meta">No clip loaded</span>
            )}
          </div>
          {clip && phase !== 'recording' && (
            <p className="scope-window">
              Model hears {formatSeconds(windowStart)} to {formatSeconds(windowEnd)}
              {clip.duration > WINDOW_SECONDS && <span className="scope-hint">Drag the bracket to choose another part</span>}
            </p>
          )}
        </div>

        <div className="scope-body">
          <Scope
            clip={clip}
            image={image}
            windowStart={windowStart}
            onWindowChange={setWindowStart}
            playheadRef={playheadRef}
            playing={playing}
            live={live}
            busy={busy}
          />
          {!clip && phase !== 'recording' && (
            <div className="scope-empty">
              <p className="scope-empty-title">{phase === 'decoding' ? 'Reading audio…' : 'Drop an audio file here'}</p>
              <p className="scope-empty-sub">WAV, MP3, FLAC, OGG or M4A, up to {MAX_SECONDS / 60} minutes. Nothing leaves your computer.</p>
            </div>
          )}
        </div>

        <div className="scope-controls">
          <div className="scope-inputs">
            <input
              ref={fileRef}
              type="file"
              accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
              className="visually-hidden"
              tabIndex={-1}
              aria-label="Audio file"
              onChange={(e) => {
                onFiles(e.target.files)
                e.target.value = ''
              }}
            />
            <button type="button" className="btn btn-scope" onClick={() => fileRef.current?.click()} disabled={busy}>
              <IconFile /> {clip ? 'Choose another file' : 'Choose a file'}
            </button>
            {phase === 'recording' ? (
              <button type="button" className="btn btn-scope is-recording" onClick={() => recorder.current?.stop()}>
                <IconStop /> Stop recording
              </button>
            ) : (
              <button type="button" className="btn btn-scope" onClick={startRecording} disabled={busy}>
                <IconMic /> Record
              </button>
            )}
            {clip && phase !== 'recording' && (
              <button type="button" className="btn btn-scope" onClick={togglePlay} disabled={phase === 'decoding'} aria-pressed={playing}>
                {playing ? <IconStop /> : <IconPlay />} {playing ? 'Stop' : 'Play window'}
              </button>
            )}
          </div>
          <button type="button" className="btn btn-check" onClick={runAnalysis} disabled={!clip || !ready || busy}>
            {phase === 'analyzing' && <span className="spinner" aria-hidden="true" />}
            {checkLabel}
          </button>
        </div>
        {error && (
          <p className="scope-error" role="alert">
            {error}
          </p>
        )}
      </div>

      <div id="verdict" className="analyzer-result">
        {latest ? (
          <Verdict key={latest.id} run={latest} />
        ) : (
          <p className="analyzer-empty">
            {ready
              ? 'Load a clip and press Check this clip. The verdict and a layer-by-layer breakdown appear here.'
              : status.message}
          </p>
        )}
      </div>

      {runs.length > 0 && <SessionLog runs={runs} onClear={() => setRuns([])} />}
    </section>
  )
}

const IconFile = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5M9 14l3-3 3 3M12 11v6" />
  </svg>
)
const IconMic = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </svg>
)
const IconPlay = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" />
  </svg>
)
const IconStop = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
)
