import { useState } from 'react'
import { REPO_URL } from './Header'
import './Footer.css'

const COMMAND = 'git clone https://github.com/Anumay1231/B.S-Detector && cd B.S-Detector && python run.py'

const LINKS = [
  { href: `${REPO_URL}/archive/refs/heads/main.zip`, label: 'Download the ZIP' },
  { href: REPO_URL, label: 'Source on GitHub' },
  { href: `${REPO_URL}/blob/main/docs/calibration.md`, label: 'Speaker calibration notes' },
  { href: `${REPO_URL}/blob/main/docs/EXPERIMENT_PROTOCOL_AND_RESULTS.md`, label: 'Deepfake experiment protocol' },
]

const CREDITS = [
  { href: 'https://huggingface.co/sukhdeveyash/XLS-R-SLS-Deepfake-Detection', label: 'SLS checkpoint' },
  { href: 'https://huggingface.co/facebook/wav2vec2-xls-r-300m', label: 'XLS-R 300M' },
  { href: 'https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb', label: 'ECAPA-TDNN' },
  { href: 'https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof', label: 'SEA-Spoof' },
  { href: 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/', label: 'VoxCeleb1' },
  { href: '/samples/CREDITS.md', label: 'Sample credits' },
]

export function Footer() {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(COMMAND)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      /* clipboard blocked: the command is still selectable */
    }
  }

  return (
    <footer className="run band-fog" id="run">
      <div className="page run-inner">
        <div className="run-main reveal">
          <h2>
            Run it <em>yourself</em>
          </h2>
          <p>
            Needs Python 3.10 or newer. The first run installs PyTorch and SpeechBrain and downloads about 1.4 GB of model weights. Or download
            the ZIP and double-click <code>start.bat</code> or <code>start.command</code>.
          </p>
          <div className="cmd">
            <code>{COMMAND}</code>
            <button type="button" className="pill pill-filled pill-sm" onClick={copy}>
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>
        <ul className="run-links reveal">
          {LINKS.map((l) => (
            <li key={l.href}>
              <a className="text-link" href={l.href} target="_blank" rel="noreferrer">
                {l.label} <span aria-hidden="true">→</span>
              </a>
            </li>
          ))}
        </ul>
      </div>
      <div className="page run-foot">
        <p>B.S. Detector is a research tool. Its verdicts are evidence, not proof.</p>
        <ul>
          {CREDITS.map((c) => (
            <li key={c.href}>
              <a href={c.href} target="_blank" rel="noreferrer">
                {c.label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </footer>
  )
}
