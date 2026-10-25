import { useEffect, useState } from 'react'
import { fetchStatus, type Status } from './lib/api'
import { Header } from './components/Header'
import { Analyzer } from './components/Analyzer'
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
      timer = window.setTimeout(poll, s.state === 'ready' ? 15000 : 2000)
    }
    void poll()
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [])
  return status
}

export default function App() {
  const status = useStatus()
  return (
    <>
      <a className="skip-link" href="#analyze">
        Skip to the analyzer
      </a>
      <Header status={status} />
      <main>
        <Analyzer status={status} />
        <Findings />
        <Method />
      </main>
      <Footer />
    </>
  )
}
