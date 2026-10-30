import { useEffect, useRef, useState } from 'react'
import { analyze, verifySpeaker, type Analysis, type Status, type TwoStageResult } from './api'
import { computeSpectrogram, decodeToClip, encodeWav, paintSpectrogram, windowSlice, type Clip } from './audio'
import { sampleBlob, type Manifest } from './samples'

export interface Showcase {
  clip: Clip | null
  image: HTMLCanvasElement | null
  detection: Analysis | null
  pair: TwoStageResult | null
  pairTitle: string | null
  sampleTitle: string | null
  /** 'recorded' = numbers saved in the manifest; 'live' = just computed by the local models */
  source: 'none' | 'recorded' | 'running' | 'live'
}

/**
 * The hero's floating cards show a real result from a bundled sample: the recorded
 * one first, replaced by a live run as soon as both models are ready.
 */
export function useShowcase(status: Status, manifest: Manifest | null): Showcase {
  const [show, setShow] = useState<Showcase>({
    clip: null,
    image: null,
    detection: null,
    pair: null,
    pairTitle: null,
    sampleTitle: null,
    source: 'none',
  })
  const ranLive = useRef(false)

  const real = manifest?.samples.find((s) => s.kind === 'real' && s.language.toLowerCase().startsWith('hi')) ?? manifest?.samples[0]
  const pair = manifest?.pairs.find((p) => p.expected === 'same') ?? manifest?.pairs[0]

  // wait until the page has painted and the browser is idle, so the hero never competes with first paint
  const [idle, setIdle] = useState(false)
  useEffect(() => {
    let cancel = () => {}
    const go = () => {
      // Safari has no requestIdleCallback
      if (typeof window.requestIdleCallback === 'function') {
        const id = window.requestIdleCallback(() => setIdle(true), { timeout: 2500 })
        cancel = () => window.cancelIdleCallback(id)
      } else {
        const id = setTimeout(() => setIdle(true), 1200)
        cancel = () => clearTimeout(id)
      }
    }
    if (document.readyState === 'complete') go()
    else window.addEventListener('load', go, { once: true })
    return () => {
      window.removeEventListener('load', go)
      cancel()
    }
  }, [])

  // decode the sample and show the recorded numbers
  useEffect(() => {
    if (!manifest || !real || !idle) return
    let alive = true
    void (async () => {
      try {
        const clip = await decodeToClip(await sampleBlob(real.file), real.title, 'file')
        if (!alive) return
        const recordedDet = manifest.results.analyze?.[real.id] ?? null
        const recordedPair = pair ? (manifest.results.verify?.[pair.id] ?? null) : null
        setShow((s) => ({
          ...s,
          clip,
          image: paintSpectrogram(computeSpectrogram(clip.samples), 160),
          detection: s.source === 'live' ? s.detection : recordedDet,
          pair: s.source === 'live' ? s.pair : recordedPair,
          pairTitle: pair?.title ?? null,
          sampleTitle: real.title,
          source: s.source === 'live' ? 'live' : recordedDet ? 'recorded' : 'none',
        }))
      } catch {
        /* the hero simply keeps its empty cards */
      }
    })()
    return () => {
      alive = false
    }
  }, [manifest, real, pair, idle])

  // once both models are up, re-run the sample live
  const ready = status.state === 'ready' && status.verifier?.state === 'ready'
  useEffect(() => {
    if (!ready || !manifest || !real || !show.clip || ranLive.current) return
    ranLive.current = true
    const clip = show.clip
    setShow((s) => ({ ...s, source: 'running' }))
    void (async () => {
      try {
        const detection = await analyze(encodeWav(windowSlice(clip, 0)))
        let pairResult: TwoStageResult | null = null
        const ref = pair && manifest.samples.find((s) => s.id === pair.reference)
        const tst = pair && manifest.samples.find((s) => s.id === pair.test)
        if (ref && tst) {
          const [a, b] = await Promise.all([sampleBlob(ref.file), sampleBlob(tst.file)])
          const [ca, cb] = await Promise.all([decodeToClip(a, ref.title, 'file'), decodeToClip(b, tst.title, 'file')])
          pairResult = await verifySpeaker(encodeWav(ca.samples), encodeWav(cb.samples), 0)
        }
        setShow((s) => ({ ...s, detection, pair: pairResult ?? s.pair, source: 'live' }))
      } catch {
        setShow((s) => ({ ...s, source: s.detection ? 'recorded' : 'none' }))
      }
    })()
  }, [ready, manifest, real, pair, show.clip])

  return show
}
