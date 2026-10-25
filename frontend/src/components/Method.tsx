import './Method.css'

const s = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' } as const

const NODES = [
  {
    stage: 1,
    title: 'Audio',
    spec: '16 kHz mono',
    body: 'Your browser converts the clip and cuts the window you chose.',
    glyph: <path d="M4 20h3l2-8 3 16 3-22 3 18 2-8 3 4h5" {...s} />,
  },
  {
    stage: 1,
    title: 'Spectrogram',
    spec: '4.04 s window',
    body: '64,600 samples, looped if the clip is shorter.',
    glyph: (
      <g {...s}>
        <rect x="4" y="6" width="28" height="24" rx="3" />
        <path d="M9 24v-6M13 24V12M17 24v-9M21 24V10M25 24v-5M29 24v-3" />
      </g>
    ),
  },
  {
    stage: 1,
    title: 'XLS-R',
    spec: '24 frozen layers',
    body: 'A speech model pretrained on 128 languages. None of its 300M weights change.',
    glyph: (
      <g {...s}>
        <rect x="6" y="5" width="24" height="5" rx="1.5" />
        <rect x="6" y="15.5" width="24" height="5" rx="1.5" />
        <rect x="6" y="26" width="24" height="5" rx="1.5" />
      </g>
    ),
  },
  {
    stage: 1,
    title: 'Layer gate',
    spec: 'Sensitive Layer Selection, 23.4M trained',
    body: 'A learned gate weighs each layer, then the blend is pooled and classified.',
    glyph: <path d="M6 8h10l8 10-8 10H6M24 18h7" {...s} />,
  },
  {
    stage: 1,
    title: 'Verdict',
    spec: 'Generated if p ≥ 0.5',
    body: 'Two probabilities: generated or real.',
    glyph: (
      <g {...s}>
        <circle cx="18" cy="18" r="12" />
        <path d="M12 18.5l4 4 8-9" />
      </g>
    ),
  },
  {
    stage: 2,
    title: 'Voiceprint',
    spec: 'ECAPA-TDNN, 192 numbers',
    body: 'One for the voice to check and one for the known voice, from whole clips.',
    glyph: <path d="M5 18c3-9 5-9 7 0s4 9 6 0 4-9 6 0 4 9 7 0" {...s} />,
  },
  {
    stage: 2,
    title: 'Match score',
    spec: 'Same speaker if ≥ 0.253',
    body: 'Cosine similarity against a cut-off calibrated on VoxCeleb1.',
    glyph: (
      <g {...s}>
        <path d="M5 26h26" />
        <path d="M14 26a9 9 0 0 1 18 0" opacity="0.4" />
        <path d="M18 26l7-9" />
      </g>
    ),
  },
]

const LIMITS = [
  'Both stages were tested on clean recordings. Phone-call audio (8 kHz, compressed) has not been evaluated.',
  'Five generators made the benchmark fakes. A new one can fool it in ways these numbers do not show.',
  'The deepfake check hears 4.04 seconds at a time. Check several windows of a long clip.',
  'The app uses the Hindi head trained on 500 clips per class when models/sls_head_n500_seed2_best.pth is present. Without it, it falls back to the English checkpoint, which misses most Hindi fakes.',
  'Nobody has tested the speaker cut-off on Hindi or Hinglish yet. It was calibrated on VoxCeleb1, mostly English interviews.',
  'A verdict is evidence, not proof.',
]

export function Method() {
  return (
    <section className="method section band-fog" id="method" aria-labelledby="method-title">
      <div className="page method-inner">
        <header className="method-head reveal">
          <h2 id="method-title">
            From waveform to <em>verdict</em>
          </h2>
          <p>Everything runs on your machine. The audio goes from this page to the Python server on localhost and nowhere else.</p>
        </header>

        <figure className="flowchart reveal" aria-label="Signal flow through both stages">
          <div className="flowchart-stages" aria-hidden="true">
            <span style={{ gridColumn: '1 / span 5' }}>Stage 1, is the voice real?</span>
            <span style={{ gridColumn: '6 / span 2' }}>Stage 2, is it the same person?</span>
          </div>
          <ol className="flowchart-nodes">
            {NODES.map((n, i) => (
              <li key={n.title} className={`fc-node${n.stage === 2 ? ' is-stage2' : ''}${i === 5 ? ' after-lock' : ''}`} style={{ '--i': i } as React.CSSProperties}>
                <svg viewBox="0 0 36 36" width="36" height="36" aria-hidden="true" className="fc-glyph">
                  {n.glyph}
                </svg>
                <span className="fc-title">{n.title}</span>
                <span className="fc-spec">{n.spec}</span>
                <span className="fc-body">{n.body}</span>
              </li>
            ))}
          </ol>
          <figcaption className="interlock">
            <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
              <rect x="5" y="11" width="14" height="10" rx="2.5" fill="currentColor" />
              <path className="shackle" d="M8 11V8a4 4 0 0 1 8 0v3" fill="none" stroke="currentColor" strokeWidth="2" />
            </svg>
            <p>
              <strong>The interlock.</strong> If the verdict says generated, the voiceprint step never runs. B.S. Detector never confirms who is
              speaking in a voice that may be a clone.
            </p>
          </figcaption>
        </figure>

        <div className="limits reveal">
          <h3 className="serif">Before you rely on it</h3>
          <ul>
            {LIMITS.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}
