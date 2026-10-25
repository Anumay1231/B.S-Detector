import { useState } from 'react'
import { benchmark, regimes } from '../data/research'
import './DotGrid.css'

const COLS = 20
const ROWS = 22
// Truth colours: real voices in ink and grey, the five generators in sienna tints.
const SLICE_COLORS = ['#17191c', '#8a8d95', '#5d2a1a', '#7d4a39', '#9e6e5e', '#b98c7c', '#d4b2a5']

type View = 'source' | 'zero' | 'n50'

interface Dot {
  real: boolean
  slice: number
  /** position within its class, 0..199, used to decide which dots a count covers */
  k: number
}

const dots: Dot[] = []
benchmark.forEach((s, slice) => {
  for (let i = 0; i < s.count; i++) {
    const real = s.kind === 'genuine'
    dots.push({ real, slice, k: dots.filter((d) => d.real === real).length })
  }
})

interface Placed {
  col: number
  row: number
  color: string
  ring: boolean
}

/** Fill a pile row by row starting at `row0`; returns the next free row. */
function pile(items: { i: number; color: string; ring: boolean }[], row0: number, out: Placed[]) {
  items.forEach((it, n) => {
    out[it.i] = { col: n % COLS, row: row0 + Math.floor(n / COLS), color: it.color, ring: it.ring }
  })
  return row0 + Math.ceil(items.length / COLS)
}

function layout(view: View): { placed: Placed[]; piles: { label: string; count: number; row: number }[] } {
  const placed: Placed[] = []
  if (view === 'source') {
    const real = dots.map((d, i) => ({ d, i })).filter((x) => x.d.real)
    const fake = dots.map((d, i) => ({ d, i })).filter((x) => !x.d.real)
    pile(real.map((x) => ({ i: x.i, color: SLICE_COLORS[x.d.slice], ring: false })), 0, placed)
    pile(fake.map((x) => ({ i: x.i, color: SLICE_COLORS[x.d.slice], ring: false })), 11, placed)
    return {
      placed,
      piles: [
        { label: 'Real voices', count: 200, row: 0 },
        { label: 'Generated', count: 200, row: 11 },
      ],
    }
  }
  const r = view === 'zero' ? regimes[0] : regimes[2]
  const realPassed: number[] = []
  const realFlagged: number[] = []
  const fakeCaught: number[] = []
  const fakeMissed: number[] = []
  dots.forEach((d, i) => {
    if (d.real) (d.k < r.genuinePassed ? realPassed : realFlagged).push(i)
    else (d.k < r.fakesCaught ? fakeCaught : fakeMissed).push(i)
  })
  const ink = SLICE_COLORS[0]
  const sienna = SLICE_COLORS[2]
  const passed = [...realPassed.map((i) => ({ i, color: ink, ring: false })), ...fakeMissed.map((i) => ({ i, color: sienna, ring: true }))]
  const flagged = [...fakeCaught.map((i) => ({ i, color: sienna, ring: false })), ...realFlagged.map((i) => ({ i, color: ink, ring: true }))]
  const next = pile(passed, 0, placed)
  const flaggedRow = next + 1
  pile(flagged, flaggedRow, placed)
  return {
    placed,
    piles: [
      { label: 'Passed as real', count: passed.length, row: 0 },
      { label: 'Flagged as generated', count: flagged.length, row: flaggedRow },
    ],
  }
}

const VIEWS: { id: View; label: string }[] = [
  { id: 'source', label: 'The test set' },
  { id: 'zero', label: 'English model' },
  { id: 'n50', label: 'After 100 Hindi clips' },
]

export function DotGrid() {
  const [view, setView] = useState<View>('zero')
  const { placed, piles } = layout(view)
  const r = view === 'zero' ? regimes[0] : regimes[2]

  const summary =
    view === 'source'
      ? '200 real Hindi voices (studio and crowdsourced) and 200 fakes, 40 from each of five generators.'
      : `${r.fakesCaught} of 200 fakes caught, ${200 - r.fakesCaught} slipped through as real. ${r.genuinePassed} of 200 real voices passed.`

  return (
    <figure className="dotgrid">
      <div className="dotgrid-tabs" role="tablist" aria-label="How to sort the 400 test clips">
        {VIEWS.map((v) => (
          <button key={v.id} type="button" role="tab" aria-selected={view === v.id} className="pill pill-sm dotgrid-tab" onClick={() => setView(v.id)}>
            {v.label}
          </button>
        ))}
      </div>

      <div className="dotgrid-field" role="img" aria-label={summary}>
        <div className="dotgrid-canvas" style={{ '--rows': ROWS } as React.CSSProperties}>
          {piles.map((p) => (
            <span key={p.label} className="dotgrid-pile" style={{ '--row': p.row } as React.CSSProperties}>
              {p.label} <span className="num">{p.count}</span>
            </span>
          ))}
          {placed.map((p, i) => (
            <span
              key={i}
              className={`dot${p.ring ? ' is-ring' : ''}`}
              style={{ '--c': p.col, '--r': p.row, '--color': p.color } as React.CSSProperties}
            />
          ))}
        </div>
      </div>

      <figcaption className="dotgrid-caption">
        <p>{summary}</p>
        {view === 'source' ? (
          <ul className="dotgrid-legend">
            {benchmark.map((s, i) => (
              <li key={s.label}>
                <span className="swatch" style={{ background: SLICE_COLORS[i] }} />
                {s.label} <span className="num tag">{s.count}</span>
              </li>
            ))}
          </ul>
        ) : (
          <ul className="dotgrid-legend">
            <li>
              <span className="swatch" style={{ background: SLICE_COLORS[0] }} /> Real voice
            </li>
            <li>
              <span className="swatch" style={{ background: SLICE_COLORS[2] }} /> Generated
            </li>
            <li>
              <span className="swatch is-ring" style={{ color: SLICE_COLORS[2] }} /> In the wrong pile
            </li>
          </ul>
        )}
        <p className="tag">Totals are from the benchmark. Which dot is which is illustrative.</p>
      </figcaption>
    </figure>
  )
}
