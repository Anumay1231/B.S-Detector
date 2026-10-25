import './Method.css'

const steps = [
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

const limits = [
  'It was tested on clean 16 kHz recordings. Phone-call audio (8 kHz, compressed) has not been evaluated.',
  'The Hindi results come from five speech generators. A new generator can fool it in ways this benchmark does not show.',
  'It only hears 4.04 seconds at a time. Check several windows of a long clip before drawing a conclusion.',
  'The app runs the English-trained checkpoint. The Hindi-adapted heads from the experiments are not bundled.',
]

export function Method() {
  return (
    <section className="method page" id="method" aria-labelledby="method-title">
      <header className="method-head">
        <h2 id="method-title">What happens to your clip</h2>
        <p>Everything runs on your machine. The audio goes from this page to the Python server on localhost and nowhere else.</p>
      </header>

      <ol className="steps">
        {steps.map((s, i) => (
          <li key={s.title} className="step">
            <span className="step-n" aria-hidden="true">
              {i + 1}
            </span>
            <h3>{s.title}</h3>
            <p className="step-spec">{s.spec}</p>
            <p className="step-body">{s.body}</p>
          </li>
        ))}
      </ol>

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
