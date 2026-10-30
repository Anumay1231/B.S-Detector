import { useEffect, useRef, useState } from 'react'
import { encodeWav } from '../lib/audio'
import { analyze, verifySpeaker, type Status } from '../lib/api'
import type { Run } from '../lib/runs'
import { SAMPLE_LABELS, sampleBlob, type Manifest } from '../lib/samples'
import { useClip } from '../lib/useClip'
import { VoiceInput } from './VoiceInput'
import { ResultCard } from './ResultCard'
import { LayerChart } from './LayerChart'
import { SessionLog } from './SessionLog'
import './Analyzer.css'

export interface SampleRequest {
  /** a sample id (stage 1 only) or a pair id (both stages) */
  id: string
  nonce: number
}

interface Props {
  status: Status
  manifest: Manifest | null
  request: SampleRequest | null
}

const noLevel = () => () => {}

export function Analyzer({ status, manifest, request }: Props) {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const autoRun = useRef(false)

  const test = useClip(noLevel)
  const known = useClip(noLevel)

  const detectorReady = status.state === 'ready'
  const verifierReady = status.verifier?.state === 'ready'
  const withSpeaker = !!known.clip
  const ready = withSpeaker ? detectorReady && verifierReady : detectorReady
  const busy = test.busy || known.busy
  const latest = runs[0]

  // the custom cursor spins while a check is listening
  useEffect(() => {
    document.documentElement.toggleAttribute('data-busy', running)
  }, [running])

  const loadSample = async (id: string, run: boolean) => {
    if (!manifest) return
    setError(null)
    const pair = manifest.pairs.find((p) => p.id === id)
    const single = manifest.samples.find((s) => s.id === id)
    try {
      if (pair) {
        const ref = manifest.samples.find((s) => s.id === pair.reference)!
        const tst = manifest.samples.find((s) => s.id === pair.test)!
        const [a, b] = await Promise.all([sampleBlob(ref.file), sampleBlob(tst.file)])
        await Promise.all([known.loadBlob(a, ref.title, 'file'), test.loadBlob(b, tst.title, 'file')])
      } else if (single) {
        known.clear()
        await test.loadBlob(await sampleBlob(single.file), single.title, 'file')
      }
      autoRun.current = run
    } catch (e) {
      setError((e as Error).message)
    }
  }

  // "Try a sample" in the hero loads the real Hindi clip and runs it
  useEffect(() => {
    if (request) void loadSample(request.id, true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request?.nonce])

  const run = async () => {
    if (!test.clip || !ready) return
    test.stopPlayback()
    known.stopPlayback()
    setError(null)
    setRunning(true)
    const base = { id: Date.now(), at: new Date(), clipName: test.clip.name, clipDuration: test.clip.duration, windowStart: 0 }
    try {
      if (known.clip) {
        const res = await verifySpeaker(encodeWav(known.clip.samples), encodeWav(test.clip.samples))
        setRuns((r) => [
          {
            ...base,
            mode: 'verify',
            analysis: res.detection,
            referenceName: known.clip!.name,
            interlocked: res.interlocked,
            interlockReason: res.interlockReason,
            referenceAnalysis: res.referenceDetection ?? null,
            verification: res.verification,
          },
          ...r,
        ])
      } else {
        const analysis = await analyze(encodeWav(test.clip.samples))
        setRuns((r) => [{ ...base, mode: 'detect', analysis }, ...r])
      }
      requestAnimationFrame(() => document.getElementById('result')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  // run once a requested sample has finished loading
  useEffect(() => {
    if (autoRun.current && test.clip && !busy && ready && !running) {
      autoRun.current = false
      void run()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [test.clip, known.clip, busy, ready])

  let label = withSpeaker ? 'Run both checks' : 'Check this voice'
  if (running) label = 'Listening…'
  else if (!ready) {
    const s = withSpeaker && detectorReady ? status.verifier?.state : status.state
    label = s === 'loading' || s === 'connecting' || s === undefined ? 'Models still loading' : 'Model unavailable'
  }

  let hint = 'The deepfake check scans the entire recording automatically.'
  if (withSpeaker) hint = 'The deepfake check scans both complete recordings first. They are compared only if both pass.'
  if (!ready) hint = withSpeaker && detectorReady ? (status.verifier?.message ?? hint) : status.message

  const chips = manifest
    ? [
        ...manifest.samples.filter((s) => s.language === 'hi').map((s) => ({ id: s.id, label: SAMPLE_LABELS[s.id] ?? s.title })),
        ...manifest.pairs.map((p) => ({ id: p.id, label: SAMPLE_LABELS[p.id] ?? p.title })),
      ]
    : []

  return (
    <section className="analyzer section" id="analyze" aria-labelledby="analyze-title">
      <div className="page analyzer-inner">
        <header className="analyzer-head reveal">
          <h2 id="analyze-title">
            Check a voice in <em>two steps</em>
          </h2>
          <p>
            Drop in a recording. B.S. Detector first asks whether the voice was generated. Add a known recording of the person it claims to be,
            and it also checks whether it is the same speaker. Nothing leaves your computer.
          </p>
        </header>

        {chips.length > 0 && (
          <div className="samples" role="group" aria-label="Bundled samples">
            <span className="tag">Or try a sample</span>
            {chips.map((c) => (
              <button key={c.id} type="button" className="pill pill-ghost pill-sm" onClick={() => void loadSample(c.id, false)} disabled={running || busy}>
                {c.label}
              </button>
            ))}
          </div>
        )}

        <div className="flow">
          <VoiceInput
            slot={test}
            title="Voice to check"
            helper="The recording you are unsure about. WAV, MP3, FLAC, OGG or M4A, up to 2 minutes."
            placeholder="Drop a clip here, choose a file or record"
            locked={running || known.busy}
          />
          <div className="flow-link" aria-hidden="true">
            <svg viewBox="0 0 40 40" width="40" height="40">
              <path d="M8 20h22m-7-7 7 7-7 7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <VoiceInput
            slot={known}
            title="Known voice"
            helper="Optional. A real recording of the person the voice claims to be. 10 to 15 seconds of speech works best."
            placeholder="Add a known voice for the speaker check"
            locked={running || test.busy}
            optional
          />
        </div>

        <div className="actions">
          <button type="button" className="pill pill-filled" onClick={run} disabled={!test.clip || !ready || running || busy}>
            {running && (
              <span className="eq" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
              </span>
            )}
            {label}
          </button>
          <p className="actions-hint">{hint}</p>
          {error && (
            <p className="actions-error" role="alert">
              {error}
            </p>
          )}
        </div>

        <div id="result" className="result-area">
          {latest ? (
            <>
              <ResultCard key={latest.id} run={latest} />
              <div className="layers-card">
                <div className="layers-card-copy">
                  <h3>Where the deepfake check listened</h3>
                  <p>
                    Each of XLS-R's 24 layers has a learned gate. Early layers carry acoustic detail, later ones carry speech sounds and words.
                    The darkest bars are the three it leaned on most.
                  </p>
                </div>
                <LayerChart weights={latest.analysis.layerWeights} tone={latest.analysis.verdict === 'SPOOF' ? 'synthetic' : 'genuine'} />
              </div>
            </>
          ) : (
            <p className="result-empty">The result appears here, with both stages side by side.</p>
          )}
        </div>

        {runs.length > 0 && <SessionLog runs={runs} onClear={() => setRuns([])} />}
      </div>
    </section>
  )
}
