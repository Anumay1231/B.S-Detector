import { useEffect, useRef } from 'react'
import type { Showcase } from '../lib/useShowcase'
import { topLayers } from '../lib/layers'
import './Hero.css'

interface Props {
  show: Showcase
  onTrySample: () => void
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`

function SourceTag({ show }: { show: Showcase }) {
  const text = show.source === 'live' ? 'Live result' : show.source === 'running' ? 'Running live…' : show.source === 'recorded' ? 'Recorded result' : 'Sample'
  return <span className="tag">{text}</span>
}

/** Spectrogram fragment of the bundled sample, with a playhead while the live run is going. */
function SpectrogramArtifact({ show }: { show: Showcase }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || !show.image) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    c.width = Math.round(c.clientWidth * dpr)
    c.height = Math.round(c.clientHeight * dpr)
    const ctx = c.getContext('2d')!
    ctx.imageSmoothingEnabled = true
    ctx.drawImage(show.image, 0, 0, c.width, c.height)
  }, [show.image])

  return (
    <figure className="artifact artifact-spec" style={{ '--i': 0 } as React.CSSProperties}>
      <figcaption className="artifact-head">
        <span className="artifact-title">{show.sampleTitle ?? 'Sample clip'}</span>
        <SourceTag show={show} />
      </figcaption>
      <div className="artifact-spec-canvas">
        <canvas ref={ref} aria-hidden="true" />
        {show.source === 'running' && <span className="artifact-playhead" aria-hidden="true" />}
      </div>
      <p className="artifact-delta">{show.clip ? `${show.clip.duration.toFixed(1)} s at 16 kHz, Hindi` : 'Loading the sample…'}</p>
    </figure>
  )
}

function Stage1Artifact({ show }: { show: Showcase }) {
  const p = show.detection?.probSpoof
  const r = 26
  const c = 2 * Math.PI * r
  return (
    <div className="artifact artifact-stat artifact-stage1" style={{ '--i': 1 } as React.CSSProperties}>
      <div className="artifact-head">
        <span className="artifact-title">Stage 1, deepfake check</span>
      </div>
      <div className="artifact-stat-row">
        <div>
          <p className="artifact-metric num">{p == null ? '–' : pct(p)}</p>
          <p className="artifact-delta">{p == null ? 'chance it was generated' : p >= 0.5 ? 'chance it was generated, flagged' : 'chance it was generated, passed'}</p>
        </div>
        <svg className="ring" viewBox="0 0 64 64" width="64" height="64" aria-hidden="true">
          <circle cx="32" cy="32" r={r} className="ring-track" />
          <circle
            cx="32"
            cy="32"
            r={r}
            className="ring-value"
            strokeDasharray={c}
            strokeDashoffset={c * (1 - (p ?? 0))}
            transform="rotate(-90 32 32)"
          />
        </svg>
      </div>
    </div>
  )
}

function LayersArtifact({ show }: { show: Showcase }) {
  const w = show.detection?.layerWeights
  const top = w ? new Set(topLayers(w)) : new Set<number>()
  const min = w ? Math.min(...w) : 0
  const span = w ? Math.max(...w) - min || 1 : 1
  return (
    <div className="artifact artifact-layers" style={{ '--i': 2 } as React.CSSProperties}>
      <div className="artifact-head">
        <span className="artifact-title">Where it listened</span>
        <span className="tag">24 layers</span>
      </div>
      <div className="mini-bars" aria-hidden="true">
        {(w ?? Array.from({ length: 24 }, () => 0)).map((x, i) => (
          <span key={i} className={top.has(i + 1) ? 'is-top' : undefined} style={{ '--h': `${w ? 12 + ((x - min) / span) * 88 : 12}%`, '--b': i } as React.CSSProperties} />
        ))}
      </div>
      <p className="artifact-delta">{w ? `Leaned on layers ${[...top].join(', ')}` : 'Layer gate weights'}</p>
    </div>
  )
}

function Stage2Artifact({ show }: { show: Showcase }) {
  const v = show.pair?.verification
  const blocked = show.pair?.interlocked
  const pos = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`
  return (
    <div className="artifact artifact-stat artifact-stage2" style={{ '--i': 3 } as React.CSSProperties}>
      <div className="artifact-head">
        <span className="artifact-title">Stage 2, speaker check</span>
      </div>
      <p className="artifact-metric num">{v ? v.score.toFixed(3) : blocked ? 'Blocked' : '–'}</p>
      <p className="artifact-delta">
        {v ? (v.outcome === 'MATCH' ? 'same speaker, cut-off 0.253' : 'different speaker, cut-off 0.253') : blocked ? 'stage 1 flagged the voice' : 'voiceprint similarity'}
      </p>
      <svg className="match-line" viewBox="0 0 200 28" preserveAspectRatio="none" aria-hidden="true">
        <path d="M2 20 H198" className="match-track" />
        {v && <path d={`M2 20 H${2 + 196 * Math.max(0, Math.min(1, v.score))}`} className="match-value" />}
        <path d={`M${2 + 196 * 0.252784} 10 V26`} className="match-cut" />
      </svg>
      {v && <span className="visually-hidden">Similarity {v.score.toFixed(3)} against a cut-off of {pos(0.252784)}.</span>}
      {show.pairTitle && <p className="tag">{show.pairTitle}</p>}
    </div>
  )
}

export function Hero({ show, onTrySample }: Props) {
  const wrap = useRef<HTMLDivElement>(null)

  // artifacts drift a few pixels as the page scrolls
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let raf = 0
    const onScroll = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => wrap.current?.style.setProperty('--drift', String(Math.min(1, window.scrollY / 600))))
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="hero-stage page" ref={wrap}>
        <div className="hero-copy">
          <h1 id="hero-title" className="hero-title">
            Is this voice <em>real</em>?
          </h1>
          <p className="hero-hindi" lang="hi">
            क्या यह आवाज़ असली है?
          </p>
          <p className="hero-sub">
            B.S. Detector checks a recording in two stages: whether the voice was generated, and whether it belongs to the person it claims
            to be. It runs on your own computer.
          </p>
          <div className="hero-actions">
            <a className="pill pill-filled" href="#analyze">
              Check a clip
            </a>
            <button type="button" className="pill pill-ghost" onClick={onTrySample}>
              Try a sample
            </button>
          </div>
        </div>

        <div className="hero-artifacts">
          <SpectrogramArtifact show={show} />
          <Stage1Artifact show={show} />
          <LayersArtifact show={show} />
          <Stage2Artifact show={show} />
        </div>
      </div>
    </section>
  )
}
