import { WINDOW_SECONDS } from '../lib/audio'
import type { Run } from '../lib/runs'
import './SessionLog.css'

function describe(r: Run): { check: string; result: string; tone: 'synthetic' | 'genuine' | 'blocked'; score: string } {
  const synth = r.analysis.verdict === 'SPOOF'
  if (r.mode === 'detect') {
    return {
      check: 'Deepfake check',
      result: synth ? 'Synthetic' : 'Genuine',
      tone: synth ? 'synthetic' : 'genuine',
      score: `${(r.analysis.probSpoof * 100).toFixed(1)}% synthetic`,
    }
  }
  if (r.interlocked || !r.verification) {
    return { check: `Against ${r.referenceName}`, result: 'Blocked, sounds synthetic', tone: 'blocked', score: `${(r.analysis.probSpoof * 100).toFixed(1)}% synthetic` }
  }
  const match = r.verification.outcome === 'MATCH'
  return {
    check: `Against ${r.referenceName}`,
    result: match ? 'Same speaker' : 'Different speaker',
    tone: match ? 'genuine' : 'synthetic',
    score: `${r.verification.score.toFixed(3)} similarity`,
  }
}

export function SessionLog({ runs, onClear }: { runs: Run[]; onClear: () => void }) {
  const exportJson = () => {
    const data = runs.map((r) => ({
      check: r.mode === 'detect' ? 'deepfake' : 'deepfake+speaker',
      clip: r.clipName,
      window: [r.windowStart, Math.min(r.clipDuration, r.windowStart + WINDOW_SECONDS)],
      verdict: r.analysis.verdict,
      probSynthetic: r.analysis.probSpoof,
      layerWeights: r.analysis.layerWeights,
      ...(r.mode === 'verify'
        ? {
            reference: r.referenceName,
            interlocked: r.interlocked,
            speaker: r.verification && {
              outcome: r.verification.outcome,
              similarity: r.verification.score,
              threshold: r.verification.threshold,
              calibrated: r.verification.isCalibrated,
            },
          }
        : {}),
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
              <th scope="col">Check</th>
              <th scope="col">Result</th>
              <th scope="col" className="num">
                Score
              </th>
              <th scope="col" className="num">
                Checked
              </th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const d = describe(r)
              return (
                <tr key={r.id}>
                  <td className="log-clip">{r.clipName}</td>
                  <td className="log-clip">{d.check}</td>
                  <td>
                    <span className="log-verdict" data-tone={d.tone}>
                      {d.result}
                    </span>
                  </td>
                  <td className="num">{d.score}</td>
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
