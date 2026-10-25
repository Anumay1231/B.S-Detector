import { useCallback, useEffect, useRef } from 'react'
import { WINDOW_SECONDS, formatSeconds, peaks, type Clip } from '../lib/audio'
import './Scope.css'

interface Props {
  clip: Clip | null
  image: HTMLCanvasElement | null
  windowStart: number
  onWindowChange: (start: number) => void
  /** seconds into the clip, or null when not playing */
  playheadRef: React.RefObject<number | null>
  playing: boolean
  /** live input while recording */
  live: AnalyserNode | null
  busy: boolean
}

const LIVE_COLUMNS = 400

export function Scope({ clip, image, windowStart, onWindowChange, playheadRef, playing, live, busy }: Props) {
  const specRef = useRef<HTMLCanvasElement>(null)
  const waveRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<{ offset: number } | null>(null)
  const liveBuf = useRef<HTMLCanvasElement | null>(null)

  const maxStart = clip ? Math.max(0, clip.duration - WINDOW_SECONDS) : 0
  const windowLen = clip ? Math.min(WINDOW_SECONDS, clip.duration) : WINDOW_SECONDS

  const fit = (c: HTMLCanvasElement) => {
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const w = Math.round(c.clientWidth * dpr)
    const h = Math.round(c.clientHeight * dpr)
    if (c.width !== w || c.height !== h) {
      c.width = w
      c.height = h
    }
    return { w, h, dpr }
  }

  const draw = useCallback(() => {
    const c = specRef.current
    if (!c) return
    const { w, h, dpr } = fit(c)
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, w, h)

    if (live && liveBuf.current) {
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(liveBuf.current, 0, 0, w, h)
      return
    }
    if (!clip || !image) return

    ctx.imageSmoothingEnabled = true
    ctx.drawImage(image, 0, 0, w, h)

    const pxPerSec = w / clip.duration
    const x0 = windowStart * pxPerSec
    const x1 = (windowStart + windowLen) * pxPerSec

    // dim everything the model won't hear
    ctx.fillStyle = 'rgba(8, 10, 20, 0.66)'
    ctx.fillRect(0, 0, x0, h)
    ctx.fillRect(x1, 0, w - x1, h)

    // window brackets
    ctx.strokeStyle = 'rgba(252, 253, 191, 0.95)'
    ctx.lineWidth = 2 * dpr
    const tick = 10 * dpr
    ctx.beginPath()
    for (const x of [x0 + ctx.lineWidth / 2, x1 - ctx.lineWidth / 2]) {
      const dir = x < (x0 + x1) / 2 ? 1 : -1
      ctx.moveTo(x + dir * tick, ctx.lineWidth / 2)
      ctx.lineTo(x, ctx.lineWidth / 2)
      ctx.lineTo(x, h - ctx.lineWidth / 2)
      ctx.lineTo(x + dir * tick, h - ctx.lineWidth / 2)
    }
    ctx.stroke()

    const ph = playheadRef.current
    if (ph != null) {
      const x = ph * pxPerSec
      ctx.fillStyle = '#fcfdbf'
      ctx.fillRect(x - dpr, 0, 2 * dpr, h)
    }
  }, [clip, image, windowStart, windowLen, playheadRef, live])

  const drawWave = useCallback(() => {
    const c = waveRef.current
    if (!c) return
    const { w, h } = fit(c)
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, w, h)
    if (!clip) return
    const buckets = Math.max(1, Math.floor(w / 2))
    const pk = peaks(clip.samples, buckets)
    let norm = 0
    for (let i = 0; i < pk.length; i++) norm = Math.max(norm, Math.abs(pk[i]))
    norm = norm || 1
    const mid = h / 2
    const x0 = (windowStart / clip.duration) * w
    const x1 = ((windowStart + windowLen) / clip.duration) * w
    for (let b = 0; b < buckets; b++) {
      const x = (b / buckets) * w
      const inside = x >= x0 && x <= x1
      ctx.fillStyle = inside ? '#fe9f6d' : 'rgba(141, 148, 171, 0.45)'
      const lo = (pk[b * 2] / norm) * mid * 0.92
      const hi = (pk[b * 2 + 1] / norm) * mid * 0.92
      ctx.fillRect(x, mid - hi, Math.max(1, (w / buckets) * 0.6), Math.max(1, hi - lo))
    }
  }, [clip, windowStart, windowLen])

  useEffect(() => {
    draw()
    drawWave()
    const ro = new ResizeObserver(() => {
      draw()
      drawWave()
    })
    if (wrapRef.current) ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [draw, drawWave])

  // playhead animation
  useEffect(() => {
    if (!playing) {
      draw()
      return
    }
    let raf = 0
    const tick = () => {
      draw()
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, draw])

  // live scrolling spectrogram while recording
  useEffect(() => {
    if (!live) {
      liveBuf.current = null
      return
    }
    const buf = document.createElement('canvas')
    buf.width = LIVE_COLUMNS
    buf.height = 128
    const bctx = buf.getContext('2d')!
    bctx.fillStyle = '#000004'
    bctx.fillRect(0, 0, buf.width, buf.height)
    liveBuf.current = buf
    const bins = new Uint8Array(live.frequencyBinCount)
    const col = bctx.createImageData(1, buf.height)
    const lut = ['#000004', '#3b0f70', '#8c2981', '#de4968', '#fe9f6d', '#fcfdbf'].map((hex) => [
      parseInt(hex.slice(1, 3), 16),
      parseInt(hex.slice(3, 5), 16),
      parseInt(hex.slice(5, 7), 16),
    ])
    const color = (v: number) => {
      const t = v * (lut.length - 1)
      const k = Math.min(lut.length - 2, Math.floor(t))
      const u = t - k
      return lut[k].map((c, i) => c + (lut[k + 1][i] - c) * u)
    }
    let raf = 0
    const tick = () => {
      live.getByteFrequencyData(bins)
      bctx.drawImage(buf, -2, 0)
      for (let y = 0; y < buf.height; y++) {
        const frac = 1 - y / (buf.height - 1)
        const bin = Math.min(bins.length - 1, Math.floor(Math.pow(frac, 1.8) * bins.length * 0.5))
        const [r, g, b] = color(Math.pow(bins[bin] / 255, 1.4))
        col.data[y * 4] = r
        col.data[y * 4 + 1] = g
        col.data[y * 4 + 2] = b
        col.data[y * 4 + 3] = 255
      }
      bctx.putImageData(col, buf.width - 2, 0)
      bctx.putImageData(col, buf.width - 1, 0)
      draw()
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [live, draw])

  // dragging the window
  const timeAt = (clientX: number) => {
    const rect = specRef.current!.getBoundingClientRect()
    return ((clientX - rect.left) / rect.width) * (clip?.duration ?? 0)
  }
  const clamp = (s: number) => Math.min(maxStart, Math.max(0, s))

  const onPointerDown = (e: React.PointerEvent) => {
    if (!clip || maxStart === 0 || busy) return
    const t = timeAt(e.clientX)
    const inside = t >= windowStart && t <= windowStart + windowLen
    const offset = inside ? t - windowStart : windowLen / 2
    dragRef.current = { offset }
    ;(e.target as Element).setPointerCapture(e.pointerId)
    onWindowChange(clamp(t - offset))
  }
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragRef.current) return
    onWindowChange(clamp(timeAt(e.clientX) - dragRef.current.offset))
  }
  const onPointerUp = () => {
    dragRef.current = null
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!clip || maxStart === 0) return
    const step = e.shiftKey ? 1 : 0.25
    const next: Record<string, number> = {
      ArrowLeft: windowStart - step,
      ArrowDown: windowStart - step,
      ArrowRight: windowStart + step,
      ArrowUp: windowStart + step,
      Home: 0,
      End: maxStart,
    }
    if (e.key in next) {
      e.preventDefault()
      onWindowChange(clamp(next[e.key]))
    }
  }

  const ticks = clip ? axisTicks(clip.duration) : []
  const movable = !!clip && maxStart > 0

  return (
    <div className="scope-display" ref={wrapRef}>
      <div className="scope-spec-wrap">
        <canvas
          ref={specRef}
          className={`scope-spec${movable ? ' is-movable' : ''}`}
          role={clip ? 'slider' : undefined}
          tabIndex={clip ? 0 : -1}
          aria-label={clip ? 'Analysis window position' : undefined}
          aria-hidden={clip ? undefined : true}
          aria-valuemin={clip ? 0 : undefined}
          aria-valuemax={clip ? Number(maxStart.toFixed(2)) : undefined}
          aria-valuenow={clip ? Number(windowStart.toFixed(2)) : undefined}
          aria-valuetext={clip ? `${formatSeconds(windowStart)} to ${formatSeconds(windowStart + windowLen)} of ${formatSeconds(clip.duration)}` : undefined}
          aria-disabled={!movable || undefined}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onKeyDown={onKeyDown}
        />
        <div className="scope-freq" aria-hidden="true">
          {[8, 4, 2, 0.5].map((k) => (
            <span key={k} style={{ bottom: `${Math.pow(k / 8, 1 / 1.8) * 100}%` }}>
              {k < 1 ? `${k * 1000} Hz` : `${k} kHz`}
            </span>
          ))}
        </div>
      </div>
      <canvas ref={waveRef} className="scope-wave" aria-hidden="true" />
      <div className="scope-axis" aria-hidden="true">
        {ticks.map((t) => (
          <span key={t} style={{ left: `${(t / (clip?.duration || 1)) * 100}%` }}>
            {t}s
          </span>
        ))}
      </div>
    </div>
  )
}

function axisTicks(duration: number): number[] {
  const step = duration <= 6 ? 1 : duration <= 15 ? 2 : duration <= 40 ? 5 : duration <= 80 ? 10 : 20
  const out: number[] = []
  for (let t = 0; t <= duration + 1e-6; t += step) out.push(Math.round(t))
  return out
}
