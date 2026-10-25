import type { Status } from '../lib/api'
import './Header.css'

// The nav itself is static HTML in index.html; only the status text is live.

export const REPO_URL = 'https://github.com/Anumay1231/B.S-Detector'

export function statusText(s: Status) {
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
