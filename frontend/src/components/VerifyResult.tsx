import type { VerifyRun } from '../lib/runs'
import { speakerBaseline } from '../data/research'
import './VerifyResult.css'

/** Similarity gauge runs 0..1; cosine scores below 0 sit at the left edge. */
const pos = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`

export function VerifyResult({ run }: { run: VerifyRun }) {
  const d = run.analysis
  const synthetic = d.verdict === 'SPOOF'
  const v = run.verification
  const match = v?.outcome === 'MATCH'
  const stage2Tone = run.interlocked ? 'blocked' : match ? 'genuine' : 'synthetic'

  return (
    <article className="twostage" aria-live="polite">
      <section className="stage" data-tone={synthetic ? 'synthetic' : 'genuine'} aria-labelledby="stage1-title">
        <p className="stage-step">Stage 1, deepfake check</p>
        <h2 className="stage-word" id="stage1-title">
          {synthetic ? 'Sounds synthetic' : 'Sounds genuine'}
        </h2>
        <p className="stage-line">
          Chance the voice to check is synthetic: <strong>{(d.probSpoof * 100).toFixed(1)}%</strong>.
        </p>
        <div className="mini-meter" aria-hidden="true">
          <span className="mini-fill" style={{ '--p': d.probSpoof } as React.CSSProperties} />
          <span className="mini-mid" />
        </div>
      </section>

      <div className="stage-gate" aria-hidden="true" data-open={!run.interlocked}>
        <svg viewBox="0 0 40 40" width="40" height="40">
          {run.interlocked ? (
            <>
              <rect x="10" y="18" width="20" height="15" rx="3" fill="currentColor" />
              <path d="M14 18v-4a6 6 0 0 1 12 0v4" fill="none" stroke="currentColor" strokeWidth="3" />
            </>
          ) : (
            <path d="M8 20h22m-8-8 8 8-8 8" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          )}
        </svg>
      </div>

      <section className="stage" data-tone={stage2Tone} aria-labelledby="stage2-title">
        <p className="stage-step">Stage 2, speaker check</p>
        {run.interlocked || !v ? (
          <>
            <h2 className="stage-word" id="stage2-title">
              Not compared
            </h2>
            <p className="stage-line">
              The voice to check sounded synthetic, so its identity was not checked. B.S. Detector never confirms who is speaking in a
              voice that may be generated.
            </p>
          </>
        ) : (
          <>
            <h2 className="stage-word" id="stage2-title">
              {match ? 'Same speaker' : 'Different speaker'}
            </h2>
            <p className="stage-line">
              Voiceprint similarity <strong>{v.score.toFixed(3)}</strong>. The calibrated cut-off is {v.threshold.toFixed(3)}; at or above it
              counts as the same speaker.
            </p>
            <div className="gauge" aria-hidden="true">
              <div className="gauge-track">
                <span className="gauge-fill" style={{ width: pos(v.score) }} />
                <span className="gauge-cut" style={{ left: pos(v.threshold) }} />
              </div>
              <div className="gauge-labels">
                <span>0</span>
                <span style={{ left: pos(v.threshold) }}>cut-off</span>
                <span>1</span>
              </div>
            </div>
          </>
        )}
      </section>

      <dl className="twostage-facts">
        <div>
          <dt>Known voice</dt>
          <dd title={run.referenceName}>{run.referenceName}</dd>
        </div>
        <div>
          <dt>Voice to check</dt>
          <dd title={run.clipName}>{run.clipName}</dd>
        </div>
        <div>
          <dt>Model runs</dt>
          <dd>
            {Math.round(d.latencyMs)} ms{v ? ` + ${Math.round(v.latencyMs)} ms` : ''}
          </dd>
        </div>
      </dl>

      <p className="verdict-caveat twostage-caveat">
        The speaker threshold comes from {speakerBaseline.trials.toLocaleString('en-US')} VoxCeleb1 trials, mostly English interviews, where
        it reached {(speakerBaseline.eer * 100).toFixed(2)}% equal error rate. It has not yet been validated on Hindi or Hinglish speech.{' '}
        <a href="#findings-speaker">See the findings</a>.
      </p>
    </article>
  )
}
