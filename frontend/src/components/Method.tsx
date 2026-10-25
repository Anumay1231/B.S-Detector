import './Method.css'

const stage1 = [
  {
    title: 'Prepare the audio',
    spec: '16 kHz mono, 64,600 samples',
    body: 'Your browser converts the clip to 16 kHz mono and cuts the 4.04-second window you chose. The server scales it to full volume and loops it if it is shorter.',
  },
  {
    title: 'Listen with XLS-R',
    spec: 'wav2vec2-xls-r-300m, frozen',
    body: 'A speech model pretrained on 128 languages turns the window into 24 layers of features, each 201 frames by 1,024 numbers. None of its 300M weights change.',
  },
  {
    title: 'Weigh the layers',
    spec: 'Selective layer summarization',
    body: 'A learned gate scores every layer, the layers are blended by those scores, then pooled and passed through two small dense layers. This head is the only part that was trained.',
  },
  {
    title: 'Decide',
    spec: 'Synthetic if p ≥ 0.5',
    body: 'The head outputs two probabilities. Anything at or above 50% synthetic is called synthetic; the gate scores become the layer chart you see with each verdict.',
  },
]

const stage2 = [
  {
    title: 'Make a voiceprint',
    spec: 'ECAPA-TDNN, 192 numbers',
    body: 'A pretrained speaker model from SpeechBrain turns each whole clip, the known voice and the voice to check, into a 192-number voiceprint. It is used exactly as published.',
  },
  {
    title: 'Compare them',
    spec: 'Cosine similarity',
    body: 'The two voiceprints are compared with cosine similarity. Recordings of the same person land near 0.6 on average; different people land near zero.',
  },
  {
    title: 'Apply the cut-off',
    spec: 'Same speaker if ≥ 0.253',
    body: 'The cut-off was calibrated on VoxCeleb1 trials, not guessed. Scores at or above it count as the same speaker, and the raw score is always shown.',
  },
]

const limits = [
  'Both stages were tested on clean recordings. Phone-call audio (8 kHz, compressed) has not been evaluated.',
  'The deepfake results come from five speech generators. A new generator can fool it in ways this benchmark does not show.',
  'The deepfake check only hears 4.04 seconds at a time. Check several windows of a long clip before drawing a conclusion.',
  'The app runs the English-trained deepfake checkpoint. The Hindi-adapted heads from the experiments are not bundled.',
  'The speaker cut-off was calibrated on VoxCeleb1, mostly English interviews. It has not been validated on Hindi or Hinglish speech.',
]

function Steps({ steps, start }: { steps: typeof stage1; start: number }) {
  return (
    <ol className="steps" start={start} style={{ '--cols': steps.length } as React.CSSProperties}>
      {steps.map((s, i) => (
        <li key={s.title} className="step">
          <span className="step-n" aria-hidden="true">
            {start + i}
          </span>
          <h4>{s.title}</h4>
          <p className="step-spec">{s.spec}</p>
          <p className="step-body">{s.body}</p>
        </li>
      ))}
    </ol>
  )
}

export function Method() {
  return (
    <section className="method page" id="method" aria-labelledby="method-title">
      <header className="method-head">
        <h2 id="method-title">What happens to your clip</h2>
        <p>Everything runs on your machine. The audio goes from this page to the Python server on localhost and nowhere else.</p>
      </header>

      <div className="method-stage">
        <h3 className="method-stage-title">
          <span>Stage 1</span> Is the voice real?
        </h3>
        <Steps steps={stage1} start={1} />
      </div>

      <div className="interlock">
        <svg viewBox="0 0 40 40" width="36" height="36" aria-hidden="true">
          <rect x="10" y="18" width="20" height="15" rx="3" fill="currentColor" />
          <path d="M14 18v-4a6 6 0 0 1 12 0v4" fill="none" stroke="currentColor" strokeWidth="3" />
        </svg>
        <p>
          <strong>The interlock.</strong> If stage 1 calls the voice synthetic, stage 2 does not run. The app never confirms who is speaking in a
          voice that may be generated, because a convincing clone would otherwise pass as the real person.
        </p>
      </div>

      <div className="method-stage">
        <h3 className="method-stage-title">
          <span>Stage 2</span> Is it the same person?
        </h3>
        <Steps steps={stage2} start={5} />
      </div>

      <div className="limits">
        <h3>Before you rely on it</h3>
        <ul>
          {limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </div>
    </section>
  )
}
