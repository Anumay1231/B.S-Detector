import { WINDOW_SECONDS, formatSeconds } from '../lib/audio'
import type { DetectRun } from '../lib/runs'
import { regimes } from '../data/research'
import { topLayers } from '../lib/layers'
import { LayerChart } from './LayerChart'
import './Verdict.css'

const bandOf = (layer: number) => (layer <= 8 ? 'acoustic detail' : layer <= 16 ? 'speech sounds' : 'words and language')

function listLayers(ls: number[]) {
  return ls.length > 1 ? `${ls.slice(0, -1).join(', ')} and ${ls[ls.length - 1]}` : String(ls[0])
}

export function Verdict({ run }: { run: DetectRun }) {
  const { analysis: a } = run
  const synthetic = a.verdict === 'SPOOF'
  const tone = synthetic ? 'synthetic' : 'genuine'
  const p = a.probSpoof
  const close = p > 0.35 && p < 0.65
  const top = topLayers(a.layerWeights)
  const bands = top.map(bandOf)
  const mainBand = bands.sort((x, y) => bands.filter((b) => b === y).length - bands.filter((b) => b === x).length)[0]
  const end = Math.min(run.clipDuration, run.windowStart + WINDOW_SECONDS)
  const zero = regimes[0]

  return (
    <article className="verdict" data-tone={tone} aria-live="polite">
      <div className="verdict-main">
        <h2 className="verdict-word">{synthetic ? 'Sounds synthetic' : 'Sounds genuine'}</h2>
        <p className="verdict-line">
          The model puts the chance that this window is synthetic at <strong>{(p * 100).toFixed(1)}%</strong>.
          {close && ' That is close to the 50% line, so treat this one as undecided.'}
        </p>

        <div className="meter" aria-hidden="true">
          <div className="meter-track">
            <span className="meter-fill" style={{ '--p': p } as React.CSSProperties} />
            <span className="meter-mid" />
          </div>
          <div className="meter-labels">
            <span>Genuine</span>
            <span>50%</span>
            <span>Synthetic</span>
          </div>
        </div>

        <dl className="verdict-facts">
          <div>
            <dt>Window heard</dt>
            <dd>
              {formatSeconds(run.windowStart)} to {formatSeconds(end)}
              {run.clipDuration < WINDOW_SECONDS && ', looped to fill 4.04 s'}
            </dd>
          </div>
          <div>
            <dt>Model run</dt>
            <dd>{a.latencyMs < 1000 ? `${Math.round(a.latencyMs)} ms` : `${(a.latencyMs / 1000).toFixed(1)} s`}</dd>
          </div>
          <div>
            <dt>Device</dt>
            <dd>{a.device.toUpperCase()}</dd>
          </div>
        </dl>
      </div>

      <div className="verdict-layers">
        <h3>Where it listened</h3>
        <p>
          Each of XLS-R's 24 layers gets a learned gate. For this clip it leaned most on layers {listLayers(top)}, mostly in the{' '}
          {mainBand} range.
        </p>
        <LayerChart weights={a.layerWeights} tone={tone} />
      </div>

      <p className="verdict-caveat">
        This build runs the English-trained checkpoint. On our Hindi test set it passed {zero.genuinePassed} of 200 real voices but caught only{' '}
        {zero.fakesCaught} of 200 fakes, so on Hindi speech a “genuine” verdict is weak evidence. <a href="#findings">See the findings</a>.
      </p>
    </article>
  )
}
