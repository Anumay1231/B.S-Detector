import { useEffect, useState } from 'react'
import type { Status } from '../lib/api'
import './Header.css'

export const REPO_URL = 'https://github.com/Anumay1231/B.S-Detector'

function statusText(s: Status) {
  const v = s.verifier?.state
  switch (s.state) {
    case 'ready':
      if (v === 'ready') return `Both models ready on ${s.device?.toUpperCase() ?? 'CPU'}`
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

/** Ready only when both stages are; any failure wins over loading. */
function overallState(s: Status) {
  const v = s.verifier?.state
  if (s.state !== 'ready') return s.state
  if (v === 'ready') return 'ready'
  return v === 'loading' || v === undefined ? 'loading' : 'error'
}

type Theme = 'light' | 'dark'

function currentTheme(): Theme {
  const set = document.documentElement.dataset.theme
  if (set === 'light' || set === 'dark') return set
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function Header({ status }: { status: Status }) {
  const [theme, setTheme] = useState<Theme>(currentTheme)

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => setTheme(currentTheme())
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    try {
      localStorage.setItem('bsd-theme', next)
    } catch {
      /* private mode: the choice just won't persist */
    }
    setTheme(next)
  }

  return (
    <header className="topbar">
      <div className="topbar-inner page">
        <a className="wordmark" href="#analyze" aria-label="B.S. Detector, back to the analyzer">
          <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
            <rect width="32" height="32" rx="8" fill="currentColor" />
            <g fill="var(--paper)">
              <rect x="6" y="13" width="2.6" height="6" rx="1.3" />
              <rect x="10.6" y="9" width="2.6" height="14" rx="1.3" />
              <rect x="15.2" y="5.5" width="2.6" height="21" rx="1.3" />
              <rect x="19.8" y="10.5" width="2.6" height="11" rx="1.3" />
              <rect x="24.4" y="14" width="2.6" height="4" rx="1.3" />
            </g>
          </svg>
          <span>B.S. Detector</span>
        </a>

        <nav className="topnav" aria-label="Sections">
          <a href="#analyze">Analyze</a>
          <a href="#findings">Findings</a>
          <a href="#method">Method</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            Source
          </a>
        </nav>

        <div className="topbar-right">
          <span className="status" data-state={overallState(status)} title={[status.message, status.verifier?.message].filter(Boolean).join(' ')} role="status">
            <span className="status-dot" aria-hidden="true" />
            {statusText(status)}
          </span>
          <button type="button" className="theme-toggle" onClick={toggle} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
            {theme === 'dark' ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  )
}
