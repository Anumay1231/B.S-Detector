import { useEffect, useState } from 'react'
import { fetchStatus, type Status } from './lib/api'
import { loadManifest, type Manifest } from './lib/samples'
import { useShowcase } from './lib/useShowcase'
import { Header } from './components/Header'
import { Hero } from './components/Hero'
import { Analyzer, type SampleRequest } from './components/Analyzer'
import { Findings } from './components/Findings'
import { Method } from './components/Method'
import { Footer } from './components/Footer'

const initial: Status = { state: 'connecting', message: 'Connecting to the local server…', device: null, checkpoint: 'v1', windowSeconds: 4.0375 }

function useStatus() {
  const [status, setStatus] = useState<Status>(initial)
  useEffect(() => {
    let timer = 0
    let alive = true
    const poll = async () => {
      const s = await fetchStatus()
      if (!alive) return
      setStatus(s)
      const settled = s.state === 'ready' && s.verifier?.state !== 'loading'
      timer = window.setTimeout(poll, settled ? 15000 : 2000)
    }
    void poll()
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [])
  return status
}

/** Sections fade up once, the first time they scroll into view. */
function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>('.reveal')
    if (!('IntersectionObserver' in window)) {
      els.forEach((el) => el.classList.add('is-visible'))
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('is-visible')
            io.unobserve(e.target)
          }
        }
      },
      { rootMargin: '0px 0px -8% 0px' },
    )
    els.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [])
}

export default function App() {
  const status = useStatus()
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [request, setRequest] = useState<SampleRequest | null>(null)
  const show = useShowcase(status, manifest)
  useReveal()

  useEffect(() => {
    void loadManifest().then(setManifest)
  }, [])

  const trySample = () => {
    const real = manifest?.samples.find((s) => s.kind === 'real' && s.language.toLowerCase().startsWith('hi')) ?? manifest?.samples[0]
    document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' })
    if (real) setRequest({ id: real.id, nonce: Date.now() })
  }

  return (
    <>
      <a className="skip-link" href="#analyze">
        Skip to the analyzer
      </a>
      <Header status={status} />
      <main>
        <Hero show={show} onTrySample={trySample} />
        <Analyzer status={status} manifest={manifest} request={request} />
        <Findings />
        <Method />
      </main>
      <Footer />
    </>
  )
}
