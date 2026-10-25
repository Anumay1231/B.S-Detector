import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import './Cursor.css'

type Mode = 'idle' | 'link' | 'listen' | 'text' | 'busy'

const LINK = 'a, button:not(:disabled), [role="tab"], label, summary, input[type="range"], .is-movable'
const LISTEN = '.artifact-spec-canvas, .scope-display, .voice-scope, .dotgrid-field'
const TEXT = 'input:not([type="range"]):not([type="file"]), textarea, [contenteditable="true"]'

/**
 * An ink dot with a lagging ring. The ring swells over things you can press, turns into
 * three listening bars over audio, and spins while a check runs. Fine pointers only;
 * touch screens and reduced motion keep the system cursor.
 */
export function Cursor() {
  const dot = useRef<HTMLDivElement>(null)
  const ring = useRef<HTMLDivElement>(null)

  const enabled =
    typeof window !== 'undefined' &&
    window.matchMedia('(pointer: fine)').matches &&
    !window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (!enabled) return
    const root = document.documentElement
    root.classList.add('has-cursor')

    let x = -100
    let y = -100
    let rx = x
    let ry = y
    let raf = 0
    let mode: Mode = 'idle'

    const setMode = (m: Mode) => {
      if (m === mode) return
      mode = m
      ring.current?.setAttribute('data-mode', m)
      dot.current?.setAttribute('data-mode', m)
    }

    const pick = (t: EventTarget | null): Mode => {
      const el = t instanceof Element ? t : null
      if (!el) return 'idle'
      if (el.closest(TEXT)) return 'text'
      if (root.hasAttribute('data-busy') && el.closest('.analyzer')) return 'busy'
      if (el.closest(LINK)) return 'link'
      if (el.closest(LISTEN)) return 'listen'
      return 'idle'
    }

    const tick = () => {
      rx += (x - rx) * 0.2
      ry += (y - ry) * 0.2
      if (ring.current) ring.current.style.transform = `translate3d(${rx}px, ${ry}px, 0)`
      raf = Math.abs(x - rx) + Math.abs(y - ry) > 0.1 ? requestAnimationFrame(tick) : 0
    }

    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== 'mouse') return
      x = e.clientX
      y = e.clientY
      if (dot.current) dot.current.style.transform = `translate3d(${x}px, ${y}px, 0)`
      root.classList.remove('cursor-out')
      setMode(pick(e.target))
      if (!raf) raf = requestAnimationFrame(tick)
    }

    const onDown = (e: PointerEvent) => {
      if (e.pointerType !== 'mouse') return
      ring.current?.classList.add('is-down')
      const ripple = document.createElement('span')
      ripple.className = 'cursor-ripple'
      ripple.style.left = `${e.clientX}px`
      ripple.style.top = `${e.clientY}px`
      ripple.addEventListener('animationend', () => ripple.remove())
      document.body.appendChild(ripple)
    }
    const onUp = () => ring.current?.classList.remove('is-down')
    const onLeave = () => root.classList.add('cursor-out')

    // the analyzer toggles data-busy while it listens; reflect it without waiting for a move
    const mo = new MutationObserver(() => setMode(pick(document.elementFromPoint(x, y))))
    mo.observe(root, { attributes: true, attributeFilter: ['data-busy'] })

    window.addEventListener('pointermove', onMove, { passive: true })
    window.addEventListener('pointerdown', onDown, { passive: true })
    window.addEventListener('pointerup', onUp, { passive: true })
    document.addEventListener('mouseleave', onLeave)
    return () => {
      root.classList.remove('has-cursor', 'cursor-out')
      cancelAnimationFrame(raf)
      mo.disconnect()
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerdown', onDown)
      window.removeEventListener('pointerup', onUp)
      document.removeEventListener('mouseleave', onLeave)
    }
  }, [enabled])

  if (!enabled) return null
  return createPortal(
    <>
      <div ref={ring} className="cursor-ring" data-mode="idle" aria-hidden="true">
        <span className="cursor-bars">
          <i />
          <i />
          <i />
        </span>
      </div>
      <div ref={dot} className="cursor-dot" data-mode="idle" aria-hidden="true" />
    </>,
    document.body,
  )
}
