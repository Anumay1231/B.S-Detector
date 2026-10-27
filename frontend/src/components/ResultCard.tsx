import { WINDOW_SECONDS, formatSeconds } from '../lib/audio'
import type { Run } from '../lib/runs'
import { speakerMeter } from '../lib/api'
import { regimes, speakerPhase9 } from '../data/research'
import './ResultCard.css'

const clamp01 = (x: number) => Math.max(0, Math.min(1, x))
const pct = (w?: { probSpoof: number } | null) => `${((w?.probSpoof ?? 0) * 100).toFixed(0)}% generated`

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
  const hindi = !!a.head?.startsWith('hindi')
  const ref = hindi ? regimes[3] : regimes[0]
  const end = Math.min(run.clipDuration, run.windowStart + WINDOW_SECONDS)

  const verify = run.mode === 'verify' ? run : null
  const v = verify?.verification ?? null
  // the interlock also trips on the known voice
  const blocked = flagged || !!verify?.interlocked

  return (
    <article className={`result${blocked ? ' is-flagged' : ''}`} aria-live="polite" aria-labelledby="rc-s1">
      <span className={`rc-stamp ${blocked ? 'is-bs' : 'is-legit'}`} aria-hidden="true">
        {blocked ? 'B.S.' : 'Legit'}
      </span>
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
            ? a.head === 'hindi-indicsynth'
              ? `This is the Hindi head trained on 500 clips per class, then fine-tuned on IndicSynth voice clones. On the team's IndicSynth test it scored 98.75%, catching 98% of XTTS-v2 clones.`
              : `This is the Hindi head trained on 500 clips per class. On the benchmark it caught ${ref.fakesCaught} of 200 fakes and passed ${ref.genuinePassed} of 200 real voices.`
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
            <p className="rc-line">
              {verify.interlockReason === 'reference'
                ? `The known voice itself sounds generated (${pct(verify.referenceAnalysis)}), so the speaker check did not run. A clone can't be the reference for the real person.`
                : 'The deepfake check flagged the voice, so the speaker check did not run. A convincing clone would otherwise pass as the real person.'}
            </p>
          </>
        ) : (
          <>
            <h3 className="rc-word serif" id="rc-s2">
              {v.outcome === 'MATCH' ? (
                <>
                  Same <em>speaker</em>
                </>
              ) : v.outcome === 'NON_MATCH' ? (
                <>
                  Different <em>speaker</em>
                </>
              ) : (
                <>
                  Not <em>sure</em>
                </>
              )}
            </h3>
            <p className="rc-line">
              {v.system === 'fused' ? 'Two speaker models together' : 'The speaker check'} score the voices{' '}
              <strong className="num">{v.score.toFixed(2)}</strong> against a cut-off of <span className="num">{v.threshold.toFixed(2)}</span>
              {v.outcome === 'UNCERTAIN'
                ? `, too close to call. Record 10 to 15 s more of each voice and check again.`
                : '.'}
            </p>
            <Stroke value={speakerMeter(v)} tick={0.5} label={`Score ${v.score.toFixed(2)}, cut-off ${v.threshold.toFixed(2)}`} />
            <p className="rc-caveat">
              {v.reliable === false
                ? `Only ${Math.min(v.referenceSeconds, v.testSeconds).toFixed(1)} s of speech in the shorter clip. Below ${v.reliableSeconds ?? 8} s the check is much less sure, so record 10 to 15 s of each voice for a firm answer.`
                : `The cut-off rises with the amount of speech and was calibrated on Hindi and Indian-English speakers. On held-out speakers with this much speech it was right on ${speakerPhase9.accuracyReliable} of the pairs it answered.`}
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
