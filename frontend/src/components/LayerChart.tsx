import { topLayers } from '../lib/layers'
import './LayerChart.css'

interface Props {
  weights: number[]
  tone: 'synthetic' | 'genuine'
}

const BANDS = [
  { from: 1, to: 8, label: 'Acoustic detail' },
  { from: 9, to: 16, label: 'Speech sounds' },
  { from: 17, to: 24, label: 'Words and language' },
]

export function LayerChart({ weights, tone }: Props) {
  const min = Math.min(...weights)
  const max = Math.max(...weights)
  const span = max - min || 1
  const top = new Set(topLayers(weights))

  return (
    <figure className="layers" data-tone={tone}>
      <div className="layers-bars" role="img" aria-label={`Gate weight per layer. Highest: layers ${[...top].join(', ')}.`}>
        {weights.map((w, i) => {
          const h = 14 + ((w - min) / span) * 86
          return (
            <div key={i} className={`layers-col${top.has(i + 1) ? ' is-top' : ''}`} style={{ '--h': `${h}%`, '--i': i } as React.CSSProperties}>
              <span className="layers-bar" />
              <span className="layers-n">{i + 1}</span>
            </div>
          )
        })}
      </div>
      <div className="layers-bands" aria-hidden="true">
        {BANDS.map((b) => (
          <span key={b.from}>
            {b.label}
            <small>
              {b.from}–{b.to}
            </small>
          </span>
        ))}
      </div>
      <figcaption className="visually-hidden">
        Gate weights for the 24 XLS-R layers, from {min.toFixed(3)} to {max.toFixed(3)}.
      </figcaption>
    </figure>
  )
}
