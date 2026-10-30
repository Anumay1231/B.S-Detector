import { ablations, bestRun, pct, regimes, speakerBaseline as sb, speakerPhase9, speakerProtocol as sp } from '../data/research'
import { REPO_URL } from './Header'
import { DotGrid } from './DotGrid'
import './Findings.css'

const DOCS = `${REPO_URL}/blob/main/docs`

/** Equal error rate per backbone setup, drawn as plain sienna strokes on a 0–30% scale. */
function AblationStrokes() {
  const max = 0.3
  return (
    <figure className="strokes">
      <ul className="strokes-list">
        {ablations.map((a) => (
          <li key={a.id} className={a.id === 'frozen' ? 'is-best' : undefined}>
            <span className="strokes-name">{a.label}</span>
            <span className="strokes-meta tag">{a.trainable} parameters trained</span>
            <svg className="strokes-line" viewBox="0 0 300 16" preserveAspectRatio="none" aria-hidden="true">
              <path d={`M2 8 H${2 + 296 * (a.eer / max)}`} />
              <circle cx={2 + 296 * (a.eer / max)} cy="8" r="3.5" />
            </svg>
            <span className="strokes-val num">{pct(a.eer)}</span>
          </li>
        ))}
      </ul>
      <figcaption className="tag">Equal error rate, mean of 3 seeds, 50 Hindi clips per class. Lower is better.</figcaption>
    </figure>
  )
}

/** Stage 2 scores on a 0–1 line: the documented averages, the highest impostor, and the cut-off. */
function ScoreLine() {
  const x = (v: number) => 12 + 476 * Math.max(0, Math.min(1, v))
  const imp = sb.calibrationImpostorMean
  const impMax = sb.calibrationImpostorMax
  const gen = sb.calibrationGenuineMean
  const cut = sb.threshold
  return (
    <figure className="scoreline">
      <svg viewBox="0 0 500 150" className="scoreline-svg" role="img" aria-label={`Different people average ${imp.toFixed(3)} similarity and reach ${impMax.toFixed(3)} at most; the same person averages ${gen.toFixed(3)}; the cut-off is ${cut.toFixed(3)}.`}>
        <path d={`M${x(0)} 92 H${x(1)}`} className="sl-axis" />
        <path d={`M${x(imp)} 92 H${x(impMax)}`} className="sl-impostor" />
        <circle cx={x(imp)} cy="92" r="6" className="sl-dot-impostor" />
        <path d={`M${x(impMax)} 84 V100`} className="sl-max" />
        <path d={`M${x(cut)} 40 V118`} className="sl-cut" />
        <circle cx={x(gen)} cy="92" r="6" className="sl-dot-genuine" />

        <text x={x(imp)} y="128" className="sl-label">different people</text>
        <text x={x(imp)} y="144" className="sl-value">{imp.toFixed(3)} average</text>
        <text x={x(impMax)} y="72" textAnchor="middle" className="sl-value">{impMax.toFixed(3)} highest</text>
        <text x={x(cut)} y="30" textAnchor="middle" className="sl-label">cut-off {cut.toFixed(3)}</text>
        <text x={x(gen)} y="128" textAnchor="middle" className="sl-label">same person</text>
        <text x={x(gen)} y="144" textAnchor="middle" className="sl-value">{gen.toFixed(3)} average</text>
        <text x={x(0)} y="78" className="sl-tick">0</text>
        <text x={x(1)} y="78" textAnchor="end" className="sl-tick">1</text>
      </svg>
      <figcaption className="tag">
        Cosine similarity on the calibration split ({sb.calibration.genuine.toLocaleString('en-US')} same-speaker and{' '}
        {sb.calibration.impostor.toLocaleString('en-US')} different-speaker pairs). Full score histograms are not in the repository.
      </figcaption>
    </figure>
  )
}

export function Findings() {
  const [zero, , n50, n500] = regimes
  return (
    <>
      <section className="chapter section band-fog" id="findings" aria-labelledby="ch1-title">
        <div className="page chapter-inner">
          <div className="feature">
            <div className="feature-copy reveal">
              <h2 id="ch1-title">
                Caught <em>{n500.fakesCaught} of 200</em> Hindi fakes after 500 training clips each
              </h2>
              <p>
                The deepfake checkpoint was trained on English. On 400 Hindi clips it had never heard, it passed {zero.genuinePassed} of 200 real
                voices but caught only {zero.fakesCaught} of 200 fakes. Retraining just its small head, with the backbone frozen, fixes most of
                that: 50 Hindi clips per class caught {n50.fakesCaught}, and 500 per class caught {n500.fakesCaught} while still passing{' '}
                {n500.genuinePassed} real voices. The best of three runs got {pct(bestRun.accuracy, 0)} right: {bestRun.fakesMissed} fakes
                missed and {bestRun.falseAlarms} false alarms out of 400.
              </p>
              <dl className="feature-stats">
                <div>
                  <dt>English model</dt>
                  <dd className="num">{pct(zero.eer)} EER</dd>
                </div>
                <div>
                  <dt>50 Hindi clips each</dt>
                  <dd className="num">{pct(n50.eer)} EER</dd>
                </div>
                <div>
                  <dt>500 Hindi clips each</dt>
                  <dd className="num">{pct(n500.eer)} EER</dd>
                </div>
              </dl>
              <a className="text-link" href={`${DOCS}/EXPERIMENT_PROTOCOL_AND_RESULTS.md`} target="_blank" rel="noreferrer">
                Read the protocol <span aria-hidden="true">→</span>
              </a>
            </div>
            <div className="feature-ui reveal">
              <DotGrid />
            </div>
          </div>

          <div className="feature feature-flip">
            <div className="feature-copy reveal">
              <h3 className="feature-h3 serif">
                Training more of the model pushed the error rate from <em>19.7% to 25.3%</em>
              </h3>
              <p>
                Letting the top layers of XLS-R learn from the same 100 clips made it overfit. With four layers unfrozen, one run called almost
                everything real and let 162 of 189 fakes through. Those runs used a separate 439-clip split of 250 real and 189 fake voices.
              </p>
            </div>
            <div className="feature-ui reveal">
              <AblationStrokes />
            </div>
          </div>

          <details className="metrics">
            <summary>All stage 1 metrics</summary>
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
                      <td className="num">
                        {pct(r.accuracy, 2)}
                        {r.accuracyStd != null && <small> ± {pct(r.accuracyStd, 2)}</small>}
                      </td>
                      <td className="num">
                        {pct(r.eer, 2)}
                        {r.eerStd != null && <small> ± {pct(r.eerStd, 2)}</small>}
                      </td>
                      <td className="num">{pct(r.precision, 2)}</td>
                      <td className="num">{pct(r.recall, 2)}</td>
                      <td className="num">{pct(r.f1, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="tag">Precision, recall and F1 treat a real voice as the positive class. Adapted rows are the mean of 3 seeds, ± one standard deviation.</p>
          </details>
        </div>
      </section>

      <section className="chapter section" id="findings-speaker" aria-labelledby="ch2-title">
        <div className="page chapter-inner">
          <div className="feature">
            <div className="feature-copy reveal">
              <h2 id="ch2-title">
                <em>{pct(sp.eer, 2)}</em> EER on VoxCeleb1
              </h2>
              <p>
                The speaker check compares voiceprints from a pretrained ECAPA-TDNN model. On the full official VoxCeleb1 list of{' '}
                {sp.trials.toLocaleString('en-US')} trials it confuses two speakers {pct(sp.eer, 2)} of the time. The team picked the cut-off on half
                the speakers and tested it on the other {sb.speakers / 2}: {pct(sb.heldOutFar, 2)} of impostors got through and{' '}
                {pct(sb.heldOutFrr, 2)} of real matches were turned away.
              </p>
              <p>
                VoxCeleb1 is mostly English interviews, and its cut-off let {speakerPhase9.oldImpostorsAccepted} of different Hindi and Indian-English
                speakers through. Recalibrated on Hindi and English speakers, with a second model and at least 8 seconds of speech per clip, the check
                is now right on{' '}
                {speakerPhase9.accuracyReliable} of the pairs it answers.
              </p>
              <a className="text-link" href={`${DOCS}/calibration.md`} target="_blank" rel="noreferrer">
                Read the calibration notes <span aria-hidden="true">→</span>
              </a>
            </div>
            <div className="feature-ui reveal">
              <div className="ui-card">
                <ScoreLine />
                <dl className="feature-stats">
                  <div>
                    <dt>Impostors let through</dt>
                    <dd className="num">{pct(sb.heldOutFar, 2)}</dd>
                  </div>
                  <div>
                    <dt>Real matches turned away</dt>
                    <dd className="num">{pct(sb.heldOutFrr, 2)}</dd>
                  </div>
                  <div>
                    <dt>ROC-AUC, full list</dt>
                    <dd className="num">{sp.rocAuc.toFixed(4)}</dd>
                  </div>
                </dl>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
