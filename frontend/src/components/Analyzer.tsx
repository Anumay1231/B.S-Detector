import { useRef, useState } from 'react'
import { encodeWav, windowSlice } from '../lib/audio'
import { analyze, verifySpeaker, type Status } from '../lib/api'
import type { Run } from '../lib/runs'
import { useClip } from '../lib/useClip'
import { ClipPanel } from './ClipPanel'
import { Verdict } from './Verdict'
import { VerifyResult } from './VerifyResult'
import { SessionLog } from './SessionLog'
import './Analyzer.css'

type Mode = 'detect' | 'verify'

const MODES: { id: Mode; title: string; detail: string }[] = [
  { id: 'detect', title: 'Check one voice', detail: 'Is it real or generated?' },
  { id: 'verify', title: 'Compare two voices', detail: 'Is it real, and is it the same person?' },
]

export function Analyzer({ status }: { status: Status }) {
  const [mode, setMode] = useState<Mode>('detect')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const titleRef = useRef<HTMLHeadingElement>(null)
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([])

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

  // The voice under test is shared by both modes, so a loaded clip carries over.
  const test = useClip(meter)
  const reference = useClip(meter)

  const detectorReady = status.state === 'ready'
  const verifierReady = status.verifier?.state === 'ready'
  const ready = mode === 'detect' ? detectorReady : detectorReady && verifierReady
  const clipsBusy = test.busy || (mode === 'verify' && reference.busy)
  const haveClips = mode === 'detect' ? !!test.clip : !!test.clip && !!reference.clip
  const latest = runs.find((r) => r.mode === mode)

  const switchMode = (next: Mode) => {
    if (running) return
    setError(null)
    setMode(next)
  }

  const onTabKey = (e: React.KeyboardEvent, i: number) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return
    e.preventDefault()
    const j = (i + (e.key === 'ArrowRight' ? 1 : -1) + MODES.length) % MODES.length
    switchMode(MODES[j].id)
    tabRefs.current[j]?.focus()
  }

  const run = async () => {
    if (!test.clip || !ready) return
    test.stopPlayback()
    reference.stopPlayback()
    setError(null)
    setRunning(true)
    const base = { id: Date.now(), at: new Date(), clipName: test.clip.name, clipDuration: test.clip.duration, windowStart: test.windowStart }
    try {
      if (mode === 'detect') {
        const analysis = await analyze(encodeWav(windowSlice(test.clip, test.windowStart)))
        setRuns((r) => [{ ...base, mode: 'detect', analysis }, ...r])
      } else if (reference.clip) {
        const res = await verifySpeaker(encodeWav(reference.clip.samples), encodeWav(test.clip.samples), test.windowStart)
        setRuns((r) => [
          {
            ...base,
            mode: 'verify',
            analysis: res.detection,
            referenceName: reference.clip!.name,
            interlocked: res.interlocked,
            verification: res.verification,
          },
          ...r,
        ])
      }
      requestAnimationFrame(() => document.getElementById('verdict')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  let label = mode === 'detect' ? 'Check this clip' : 'Run both checks'
  if (running) label = 'Listening…'
  else if (!ready) {
    const s = mode === 'verify' && detectorReady ? status.verifier?.state : status.state
    label = s === 'loading' || s === 'connecting' || s === undefined ? 'Models still loading' : 'Model unavailable'
  }

  const hint =
    mode === 'detect'
      ? 'The deepfake check listens to the highlighted 4.04 seconds.'
      : 'The deepfake check runs first. Voices are compared only if the clip to check sounds genuine.'

  let emptyText: string
  if (mode === 'detect') emptyText = detectorReady ? 'Load a clip and press Check this clip. The verdict and a layer-by-layer breakdown appear here.' : status.message
  else if (!detectorReady) emptyText = status.message
  else if (!verifierReady) emptyText = status.verifier?.message ?? 'Connecting to the speaker model…'
  else emptyText = 'Load a known voice and a voice to check, then press Run both checks.'

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
          B.S. Detector runs two checks on a voice. A frozen XLS-R model listens for signs it was generated. If it sounds human, an ECAPA-TDNN
          speaker model checks whether it matches a voice you already know.
        </p>
      </header>

      <div className="modes" role="tablist" aria-label="What to check">
        {MODES.map((m, i) => (
          <button
            key={m.id}
            ref={(el) => {
              tabRefs.current[i] = el
            }}
            type="button"
            role="tab"
            id={`tab-${m.id}`}
            aria-selected={mode === m.id}
            aria-controls="analyzer-panel"
            tabIndex={mode === m.id ? 0 : -1}
            className="mode"
            onClick={() => switchMode(m.id)}
            onKeyDown={(e) => onTabKey(e, i)}
            disabled={running && mode !== m.id}
          >
            <span className="mode-title">{m.title}</span>
            <span className="mode-detail">{m.detail}</span>
          </button>
        ))}
      </div>

      <div id="analyzer-panel" role="tabpanel" aria-labelledby={`tab-${mode}`} className="analyzer-panel">
        {mode === 'detect' ? (
          <ClipPanel slot={test} windowed locked={running} emptyHint="Drop an audio file here" />
        ) : (
          <div className="pair">
            <ClipPanel slot={reference} label="Known voice" windowed={false} compact locked={running} emptyHint="Drop the real person’s voice" />
            <ClipPanel slot={test} label="Voice to check" windowed compact locked={running} emptyHint="Drop the voice to check" />
          </div>
        )}

        <div className="actions">
          <p className="actions-hint">{hint}</p>
          <button type="button" className="btn btn-check" onClick={run} disabled={!haveClips || !ready || running || clipsBusy}>
            {running && <span className="spinner" aria-hidden="true" />}
            {label}
          </button>
          {error && (
            <p className="actions-error" role="alert">
              {error}
            </p>
          )}
        </div>
      </div>

      <div id="verdict" className="analyzer-result">
        {latest?.mode === 'detect' && <Verdict key={latest.id} run={latest} />}
        {latest?.mode === 'verify' && <VerifyResult key={latest.id} run={latest} />}
        {!latest && <p className="analyzer-empty">{emptyText}</p>}
      </div>

      {runs.length > 0 && <SessionLog runs={runs} onClear={() => setRuns([])} />}
    </section>
  )
}
