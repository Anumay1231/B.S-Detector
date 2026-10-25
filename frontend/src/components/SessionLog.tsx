import { WINDOW_SECONDS, formatSeconds } from '../lib/audio'
import type { Run } from './Verdict'
import './SessionLog.css'

export function SessionLog({ runs, onClear }: { runs: Run[]; onClear: () => void }) {
  const exportJson = () => {
    const data = runs.map((r) => ({
      clip: r.clipName,
      window: [r.windowStart, Math.min(r.clipDuration, r.windowStart + WINDOW_SECONDS)],
      verdict: r.analysis.verdict,
      probSynthetic: r.analysis.probSpoof,
      layerWeights: r.analysis.layerWeights,
      latencyMs: r.analysis.latencyMs,
      device: r.analysis.device,
      checkedAt: r.at.toISOString(),
    }))
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `bs-detector-session-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="log" aria-labelledby="log-title">
      <div className="log-head">
        <h2 id="log-title">This session</h2>
        <p>Kept in this tab only. Closing it clears the list.</p>
        <div className="log-actions">
          <button type="button" className="btn btn-ghost" onClick={exportJson}>
            Export JSON
          </button>
          <button type="button" className="btn btn-ghost" onClick={onClear}>
            Clear
          </button>
        </div>
      </div>
      <div className="log-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Clip</th>
              <th scope="col">Window</th>
              <th scope="col">Verdict</th>
              <th scope="col" className="num">
                Chance synthetic
              </th>
              <th scope="col" className="num">
                Checked
              </th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const synth = r.analysis.verdict === 'SPOOF'
              return (
                <tr key={r.id}>
                  <td className="log-clip">{r.clipName}</td>
                  <td>
                    {formatSeconds(r.windowStart)} to {formatSeconds(Math.min(r.clipDuration, r.windowStart + WINDOW_SECONDS))}
                  </td>
                  <td>
                    <span className="log-verdict" data-tone={synth ? 'synthetic' : 'genuine'}>
                      {synth ? 'Synthetic' : 'Genuine'}
                    </span>
                  </td>
                  <td className="num">{(r.analysis.probSpoof * 100).toFixed(1)}%</td>
                  <td className="num">{r.at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
