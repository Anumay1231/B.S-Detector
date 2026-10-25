import { WINDOW_SECONDS, formatSeconds } from '../lib/audio'
import type { Run } from '../lib/runs'
import { regimes, speakerBaseline } from '../data/research'
import './ResultCard.css'

const clamp01 = (x: number) => Math.max(0, Math.min(1, x))

/** Stage 1 headline, worded so a pass on Hindi never reads as a guarantee. */
function stage1Words(p: number): [string, string] {
  if (p >= 0.65) return ['Sounds', 'generated']
  if (p >= 0.5) return ['Leans', 'generated']
  if (p > 0.35) return ['Too close to', 'call']
  return ['No sign it was', 'generated']
}

/** Meter as a single gestural stroke with one reference tick. */
function Stroke({ value, tick, label }: { value: number; tick: number; label: string }) {
  return (
    <svg className="rc-stroke" viewBox="0 0 300 24" preserveAspectRatio="none" role="img" aria-label={label}>
      <path d="M2 14 H298" className="rc-stroke-track" />
      <path d={`M2 14 H${2 + 296 * clamp01(value)}`} className="rc-stroke-value" />
      <path d={`M${2 + 296 * tick} 4 V22`} className="rc-stroke-tick" />
    </svg>
  )
}

export function ResultCard({ run }: { run: Run }) {
  const a = run.analysis
  const p = a.probSpoof
  const flagged = a.verdict === 'SPOOF'
  const [lead, em] = stage1Words(p)
  const hindi = a.head === 'hindi-n500'
  const ref = hindi ? regimes[3] : regimes[0]
  const end = Math.min(run.clipDuration, run.windowStart + WINDOW_SECONDS)

  const verify = run.mode === 'verify' ? run : null
  const v = verify?.verification ?? null

  return (
    <article className="result" aria-live="polite" aria-labelledby="rc-s1">
      <section className="rc-stage">
        <h3 className="rc-word serif" id="rc-s1">
          {lead} <em>{em}</em>
        </h3>
        <p className="rc-line">
          The deepfake check puts the chance this window was generated at <strong className="num">{(p * 100).toFixed(1)}%</strong>, so it{' '}
          {flagged ? 'flags the voice.' : 'passes the voice.'}
        </p>
        <Stroke value={p} tick={0.5} label={`${(p * 100).toFixed(1)}% chance generated, flag line at 50%`} />
        <p className="rc-caveat">
          {hindi
            ? `This is the Hindi head trained on 500 clips per class. On the benchmark it caught ${ref.fakesCaught} of 200 fakes and passed ${ref.genuinePassed} of 200 real voices.`
            : flagged
              ? `This is the English checkpoint. On the Hindi benchmark it passed ${ref.genuinePassed} of 200 real voices, so a flag is the stronger signal.`
              : `This is the English checkpoint. On the Hindi benchmark it caught only ${ref.fakesCaught} of 200 fakes, so a pass is weak evidence.`}
        </p>
      </section>

      <section className="rc-stage" aria-labelledby="rc-s2">
        {!verify ? (
          <>
            <h3 className="rc-word serif" id="rc-s2">
              Not <em>run</em>
            </h3>
            <p className="rc-line">The speaker check needs a known voice. Add one to check who is speaking.</p>
          </>
        ) : verify.interlocked || !v ? (
          <>
            <h3 className="rc-word serif" id="rc-s2">
              Not <em>compared</em>
            </h3>
            <p className="rc-line">The deepfake check flagged the voice, so the speaker check did not run. A convincing clone would otherwise pass as the real person.</p>
          </>
        ) : (
          <>
            <h3 className="rc-word serif" id="rc-s2">
              {v.outcome === 'MATCH' ? 'Same' : 'Different'} <em>speaker</em>
            </h3>
            <p className="rc-line">
              The speaker check scores the two voiceprints <strong className="num">{v.score.toFixed(3)}</strong> alike, against a cut-off of{' '}
              <span className="num">{v.threshold.toFixed(3)}</span>.
            </p>
            <Stroke value={v.score} tick={v.threshold} label={`Similarity ${v.score.toFixed(3)}, cut-off ${v.threshold.toFixed(3)}`} />
            <p className="rc-caveat">
              The cut-off comes from VoxCeleb1, mostly English interviews, where {(speakerBaseline.heldOutFar * 100).toFixed(2)}% of impostors got
              through. Nobody has tested it on Hindi or Hinglish speech yet.
            </p>
          </>
        )}
      </section>

      <dl className="rc-facts">
        <div>
          <dt>Voice checked</dt>
          <dd title={run.clipName}>{run.clipName}</dd>
        </div>
        <div>
          <dt>Window heard</dt>
          <dd className="num">
            {formatSeconds(run.windowStart)} to {formatSeconds(end)}
          </dd>
        </div>
        {verify && (
          <div>
            <dt>Known voice</dt>
            <dd title={verify.referenceName}>{verify.referenceName}</dd>
          </div>
        )}
        <div>
          <dt>Model time</dt>
          <dd className="num">
            {Math.round(a.latencyMs)} ms{v ? ` + ${Math.round(v.latencyMs)} ms` : ''} on {a.device.toUpperCase()}
          </dd>
        </div>
      </dl>
    </article>
  )
}
