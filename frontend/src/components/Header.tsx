import type { Status } from '../lib/api'
import './Header.css'

export const REPO_URL = 'https://github.com/Anumay1231/B.S-Detector'

function statusText(s: Status) {
  const v = s.verifier?.state
  switch (s.state) {
    case 'ready':
      if (v === 'ready') return `Models ready on ${s.device?.toUpperCase() ?? 'CPU'}`
      if (v === 'loading' || v === undefined) return 'Loading speaker model'
      return 'Speaker model unavailable'
    case 'loading':
    case 'connecting':
      return 'Loading models'
    case 'offline':
      return 'Server not running'
    case 'unavailable':
      return 'PyTorch missing'
    default:
      return 'Model failed to load'
  }
}

const LINKS = [
  { href: '#analyze', label: 'Analyze' },
  { href: '#findings', label: 'Findings' },
  { href: '#method', label: 'Method' },
]

export function Header({ status }: { status: Status }) {
  return (
    <header className="nav">
      <div className="nav-inner page">
        <div className="nav-left">
          <a className="logo" href="#top" aria-label="B.S. Detector, back to top">
            <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
              <rect width="32" height="32" rx="9" fill="currentColor" />
              <g fill="var(--color-blush-peach)">
                <rect x="6" y="13" width="2.6" height="6" rx="1.3" />
                <rect x="10.6" y="9" width="2.6" height="14" rx="1.3" />
                <rect x="15.2" y="5.5" width="2.6" height="21" rx="1.3" />
                <rect x="19.8" y="10.5" width="2.6" height="11" rx="1.3" />
                <rect x="24.4" y="14" width="2.6" height="4" rx="1.3" />
              </g>
            </svg>
            <span>B.S. Detector</span>
          </a>
          <span className="tag nav-status" role="status" title={[status.message, status.verifier?.message].filter(Boolean).join(' ')}>
            {statusText(status)}
          </span>
        </div>

        <nav className="nav-links" aria-label="Sections">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href}>
              {l.label}
            </a>
          ))}
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            Source
          </a>
        </nav>

        <div className="nav-right">
          <a className="text-link nav-run" href="#run">
            Run it yourself <span aria-hidden="true">→</span>
          </a>
          <a className="pill pill-filled pill-sm" href="#analyze">
            Check a clip
          </a>
        </div>
      </div>
    </header>
  )
}
