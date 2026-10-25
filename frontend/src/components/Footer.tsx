import { useState } from 'react'
import { REPO_URL } from './Header'
import './Footer.css'

const COMMAND = 'git clone https://github.com/Anumay1231/B.S-Detector && cd B.S-Detector && python run.py'

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
    <footer className="footer">
      <div className="page footer-inner">
        <div className="footer-run">
          <h2>Run it yourself</h2>
          <p>Needs Python 3.9 or newer. The first run installs PyTorch and downloads about 1.3 GB of model weights.</p>
          <div className="cmd">
            <code>{COMMAND}</code>
            <button type="button" className="btn btn-ghost" onClick={copy}>
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>

        <div className="footer-links">
          <ul>
            <li>
              <a href={REPO_URL} target="_blank" rel="noreferrer">
                Source code
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/sukhdeveyash/XLS-R-SLS-Deepfake-Detection" target="_blank" rel="noreferrer">
                SLS checkpoint
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/facebook/wav2vec2-xls-r-300m" target="_blank" rel="noreferrer">
                XLS-R 300M
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof" target="_blank" rel="noreferrer">
                SEA-Spoof dataset
              </a>
            </li>
          </ul>
          <p className="footer-note">A research tool. Its verdicts are evidence, not proof.</p>
        </div>
      </div>
    </footer>
  )
}
