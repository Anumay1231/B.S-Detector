import { ablations, benchmark, pct, regimes, speakerBaseline as sb, speakerProtocol as sp } from '../data/research'
import './Findings.css'

const SLICE_COLORS = ['var(--genuine)', 'color-mix(in srgb, var(--genuine) 45%, var(--paper))', '#3b0f70', '#8c2981', '#c33b74', '#ee6a5c', '#f9a46c']

function TestSetGrid() {
  const dots: number[] = []
  benchmark.forEach((s, i) => {
    for (let k = 0; k < s.count; k++) dots.push(i)
  })
  return (
    <figure className="testset">
      <div className="testset-grid" role="img" aria-label="400 test clips: 200 genuine, 200 synthetic from five generators, 40 each.">
        {dots.map((d, i) => (
          <span key={i} style={{ background: SLICE_COLORS[d] }} />
        ))}
      </div>
      <figcaption className="testset-legend">
        <div>
          <p className="testset-group">200 real voices</p>
          <ul>
            {benchmark.slice(0, 2).map((s, i) => (
              <li key={s.label}>
                <span className="swatch" style={{ background: SLICE_COLORS[i] }} />
                <span className="legend-name">{s.label}</span>
                <span className="legend-detail">{s.detail}</span>
                <span className="legend-count">{s.count}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="testset-group">200 fakes, 40 from each generator</p>
          <ul>
            {benchmark.slice(2).map((s, i) => (
              <li key={s.label}>
                <span className="swatch" style={{ background: SLICE_COLORS[i + 2] }} />
                <span className="legend-name">{s.label}</span>
                <span className="legend-detail">{s.detail}</span>
                <span className="legend-count">{s.count}</span>
              </li>
            ))}
          </ul>
        </div>
      </figcaption>
    </figure>
  )
}

function CatchChart() {
  return (
    <figure className="catch">
      <div className="catch-key" aria-hidden="true">
        <span>
          <i className="mark mark-genuine" /> Real voices passed
        </span>
        <span>
          <i className="mark mark-synthetic" /> Fakes caught
        </span>
        <span className="catch-scale">out of 200 each</span>
      </div>
      <ol className="catch-rows">
        {regimes.map((r) => (
          <li key={r.id} className="catch-row">
            <div className="catch-label">
              <span className="catch-name">{r.label}</span>
              <span className="catch-shots">{r.shots}</span>
            </div>
            <div className="catch-track" aria-hidden="true">
              <span className="catch-span" style={{ left: `${(Math.min(r.fakesCaught, r.genuinePassed) / 200) * 100}%`, right: `${100 - (Math.max(r.fakesCaught, r.genuinePassed) / 200) * 100}%` }} />
              <i className="mark mark-genuine" style={{ left: `${(r.genuinePassed / 200) * 100}%` }} />
              <i className="mark mark-synthetic" style={{ left: `${(r.fakesCaught / 200) * 100}%` }} />
            </div>
            <div className="catch-nums">
              <span className="num-synthetic">
                <strong>{r.fakesCaught}</strong> caught
              </span>
              <span className="num-genuine">
                <strong>{r.genuinePassed}</strong> passed
              </span>
            </div>
            <span className="visually-hidden">
              {r.label}: caught {r.fakesCaught} of 200 fakes and passed {r.genuinePassed} of 200 real voices. Equal error rate {pct(r.eer)}.
            </span>
          </li>
        ))}
      </ol>
      <div className="catch-axis" aria-hidden="true">
        <span>0</span>
        <span>100</span>
        <span>200</span>
      </div>
    </figure>
  )
}

function Matrix({ m }: { m: [[number, number], [number, number]] }) {
  const [[ss, sg], [gs, gg]] = m
  return (
    <table className="matrix">
      <caption className="visually-hidden">Confusion matrix for one seed</caption>
      <thead>
        <tr>
          <td />
          <th scope="col">Called fake</th>
          <th scope="col">Called real</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">Fake</th>
          <td className="hit">{ss}</td>
          <td className="miss">{sg}</td>
        </tr>
        <tr>
          <th scope="row">Real</th>
          <td className="miss-soft">{gs}</td>
          <td className="hit">{gg}</td>
        </tr>
      </tbody>
    </table>
  )
}

function Ablation() {
  const maxEer = 0.3
  const top4 = ablations[2]
  const [[caught, missed]] = top4.matrix!
  return (
    <div className="ablation">
      <div className="ablation-copy">
        <h3>Training more of the model made it worse</h3>
        <p>
          We also let the top layers of XLS-R learn from the same 100 Hindi clips. With so little data they overfit. Unfreezing four layers
          produced a detector that calls almost everything real: in one run it let {missed} of {caught + missed} fakes through.
        </p>
        <p>Keeping the whole backbone frozen and training only the small head on top gave the best result.</p>
      </div>

      <div className="ablation-data">
        <ul className="eer-bars" aria-label="Equal error rate by configuration, lower is better">
          {ablations.map((a) => (
            <li key={a.id} data-best={a.id === 'frozen' || undefined}>
              <span className="eer-name">{a.label}</span>
              <span className="eer-params">{a.trainable} trainable</span>
              <span className="eer-track">
                <span className="eer-fill" style={{ width: `${(a.eer / maxEer) * 100}%` }} />
              </span>
              <span className="eer-val">{pct(a.eer)}</span>
            </li>
          ))}
        </ul>
        <p className="eer-note">Equal error rate, mean of 3 seeds. Lower is better.</p>

        <div className="matrix-wrap">
          <p className="matrix-title">Top 4 layers unfrozen, one seed</p>
          <Matrix m={top4.matrix!} />
        </div>
      </div>
    </div>
  )
}

function SpeakerFindings() {
  const mark = (x: number) => `${x * 100}%`
  return (
    <div className="speaker" id="findings-speaker">
      <div className="finding">
        <div className="finding-copy">
          <h3>On {sb.trials.toLocaleString('en-US')} VoxCeleb1 trials, it confused two speakers {pct(sb.eer, 2)} of the time.</h3>
          <p>
            The speaker check compares voiceprints from a pretrained ECAPA-TDNN model. The team chose the cut-off on half of the speakers, then
            tested it on {sb.speakers / 2} speakers it had never seen: it let {pct(sb.heldOutFar, 2)} of impostors through and turned away{' '}
            {pct(sb.heldOutFrr, 2)} of real matches.
          </p>
          <p>
            VoxCeleb1 is mostly English interviews. These numbers are an English baseline. The speaker check has not been validated on Hindi or
            Hinglish speech yet.
          </p>
        </div>

        <figure className="scoreline">
          <figcaption className="scoreline-title">Average voiceprint similarity on held-out speakers</figcaption>
          <div className="scoreline-track" aria-hidden="true">
            <span className="scoreline-zone" style={{ left: mark(sb.threshold) }} />
            <span className="scoreline-mark is-impostor" style={{ left: mark(sb.impostorMean) }} />
            <span className="scoreline-cut" style={{ left: mark(sb.threshold) }} />
            <span className="scoreline-mark is-genuine" style={{ left: mark(sb.genuineMean) }} />
          </div>
          <div className="scoreline-labels" aria-hidden="true">
            <span style={{ left: mark(sb.impostorMean) }}>
              <strong>{sb.impostorMean.toFixed(3)}</strong>
              Different people
            </span>
            <span style={{ left: mark(sb.threshold) }}>
              <strong>{sb.threshold.toFixed(3)}</strong>
              Cut-off
            </span>
            <span style={{ left: mark(sb.genuineMean) }}>
              <strong>{sb.genuineMean.toFixed(3)}</strong>
              Same person
            </span>
          </div>
          <p className="visually-hidden">
            Different people average {sb.impostorMean.toFixed(3)}, the same person averages {sb.genuineMean.toFixed(3)}, and the cut-off is{' '}
            {sb.threshold.toFixed(3)}.
          </p>

          <dl className="speaker-stats">
            <div>
              <dt>Impostors let through</dt>
              <dd>{pct(sb.heldOutFar, 2)}</dd>
            </div>
            <div>
              <dt>Real matches turned away</dt>
              <dd>{pct(sb.heldOutFrr, 2)}</dd>
            </div>
            <div>
              <dt>Full official list, {sp.trials.toLocaleString('en-US')} trials</dt>
              <dd>{pct(sp.eer, 2)} EER</dd>
            </div>
          </dl>
          <p className="eer-note">
            On the full official trial list the equal error rate is {pct(sp.eer, 2)}, against about {pct(sp.publishedEer, 1)} published for the same
            model on the cleaned list. Details in docs/calibration.md.
          </p>
        </figure>
      </div>
    </div>
  )
}

export function Findings() {
  const [zero, , n50] = regimes
  return (
    <section className="findings" id="findings" aria-labelledby="findings-title">
      <div className="page">
        <header className="findings-head">
          <h2 id="findings-title">How far to trust it</h2>
          <p>
            Each stage was tested on its own. The deepfake check was trained on English and tested on 400 Hindi clips it had never heard. The
            speaker check was calibrated and tested on VoxCeleb1.
          </p>
        </header>

        <h3 className="stage-heading">
          <span>Stage 1</span> Telling a real voice from a generated one
        </h3>

        <TestSetGrid />

        <div className="finding">
          <div className="finding-copy">
            <h3>
              Out of the box it caught {zero.fakesCaught} of 200 fakes. After 100 Hindi examples, it caught {n50.fakesCaught}.
            </h3>
            <p>
              The English model almost never doubts a real voice, which is exactly the problem: it waves most Hindi fakes through too.
              Retraining just the 22.8M-parameter head on 50 clips per class trades a few real voices for far more fakes caught, and the
              equal error rate drops from {pct(zero.eer)} to {pct(n50.eer)}.
            </p>
          </div>
          <CatchChart />
        </div>

        <Ablation />

        <details className="metrics">
          <summary>All metrics</summary>
          <div className="metrics-scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col">Setup</th>
                  <th scope="col">Accuracy</th>
                  <th scope="col">EER</th>
                  <th scope="col">Precision</th>
                  <th scope="col">Recall</th>
                  <th scope="col">F1</th>
                </tr>
              </thead>
              <tbody>
                {regimes.map((r) => (
                  <tr key={r.id}>
                    <th scope="row">{r.label}</th>
                    <td>
                      {pct(r.accuracy, 2)}
                      {r.accuracyStd != null && <small> ± {pct(r.accuracyStd, 2)}</small>}
                    </td>
                    <td>
                      {pct(r.eer, 2)}
                      {r.eerStd != null && <small> ± {pct(r.eerStd, 2)}</small>}
                    </td>
                    <td>{pct(r.precision, 2)}</td>
                    <td>{pct(r.recall, 2)}</td>
                    <td>{pct(r.f1, 2)}</td>
                  </tr>
                ))}
                {ablations.slice(1).map((a) => (
                  <tr key={a.id}>
                    <th scope="row">50 per class, {a.label.toLowerCase()}</th>
                    <td>
                      {pct(a.accuracy, 2)}
                      <small> ± {pct(a.accuracyStd, 2)}</small>
                    </td>
                    <td>
                      {pct(a.eer, 2)}
                      <small> ± {pct(a.eerStd, 2)}</small>
                    </td>
                    <td colSpan={3} className="muted">
                      See outputs/ in the repository
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="metrics-note">
            Precision, recall and F1 treat “real voice” as the positive class. Adapted rows are the mean of 3 random seeds; ± is one standard
            deviation. The unfreezing runs used a 439-clip split.
          </p>
        </details>

        <h3 className="stage-heading">
          <span>Stage 2</span> Telling one speaker from another
        </h3>

        <SpeakerFindings />
      </div>
    </section>
  )
}
