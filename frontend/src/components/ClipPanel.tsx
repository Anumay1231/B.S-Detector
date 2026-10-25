import { useRef, useState } from 'react'
import { MAX_SECONDS, WINDOW_SECONDS, formatSeconds } from '../lib/audio'
import { RECORD_LIMIT, type ClipSlot } from '../lib/useClip'
import { Scope } from './Scope'

interface Props {
  slot: ClipSlot
  /** Shown above the spectrogram, e.g. "Known voice". Omitted in single-clip mode. */
  label?: string
  /** Show and let people move the 4.04 s deepfake-detection window. */
  windowed: boolean
  compact?: boolean
  /** Locks inputs while a check runs. */
  locked: boolean
  emptyHint: string
}

export function ClipPanel({ slot, label, windowed, compact, locked, emptyHint }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const { clip, phase } = slot
  const busy = slot.busy || locked
  const windowEnd = clip ? Math.min(clip.duration, slot.windowStart + WINDOW_SECONDS) : 0

  return (
    <div
      className={`scope${dragOver ? ' is-dragover' : ''}${compact ? ' is-compact' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        if (!busy) setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        if (!busy) slot.loadFiles(e.dataTransfer.files)
      }}
    >
      {label && <p className="scope-label">{label}</p>}
      <div className="scope-head">
        <div className="scope-title">
          {phase === 'recording' ? (
            <>
              <span className="rec-dot" aria-hidden="true" />
              <span>Recording</span>
              <span className="scope-meta">
                {slot.recordSeconds.toFixed(1)} of {RECORD_LIMIT} s
              </span>
            </>
          ) : clip ? (
            <>
              <span className="scope-name" title={clip.name}>
                {clip.name}
              </span>
              <span className="scope-meta">{formatSeconds(clip.duration)}</span>
            </>
          ) : (
            <span className="scope-meta">No clip loaded</span>
          )}
        </div>
        {clip && phase !== 'recording' && windowed && (
          <p className="scope-window">
            Deepfake check hears {formatSeconds(slot.windowStart)} to {formatSeconds(windowEnd)}
            {clip.duration > WINDOW_SECONDS && !compact && <span className="scope-hint">Drag the bracket to choose another part</span>}
          </p>
        )}
        {clip && phase !== 'recording' && !windowed && <p className="scope-window scope-window-quiet">Whole clip is used</p>}
      </div>

      <div className="scope-body">
        <Scope
          clip={clip}
          image={slot.image}
          windowStart={slot.windowStart}
          onWindowChange={slot.setWindowStart}
          playheadRef={slot.playheadRef}
          playing={slot.playing}
          live={slot.live}
          busy={busy}
          windowed={windowed}
        />
        {!clip && phase !== 'recording' && (
          <div className="scope-empty">
            <p className="scope-empty-title">{phase === 'decoding' ? 'Reading audio…' : emptyHint}</p>
            <p className="scope-empty-sub">
              WAV, MP3, FLAC, OGG or M4A, up to {MAX_SECONDS / 60} minutes. Nothing leaves your computer.
            </p>
          </div>
        )}
      </div>

      <div className="scope-inputs">
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
          className="visually-hidden"
          tabIndex={-1}
          aria-label={label ? `${label} audio file` : 'Audio file'}
          onChange={(e) => {
            slot.loadFiles(e.target.files)
            e.target.value = ''
          }}
        />
        <button type="button" className="btn btn-scope" onClick={() => fileRef.current?.click()} disabled={busy}>
          <IconFile /> {clip ? 'Choose another file' : 'Choose a file'}
        </button>
        {phase === 'recording' ? (
          <button type="button" className="btn btn-scope is-recording" onClick={slot.stopRecording}>
            <IconStop /> Stop recording
          </button>
        ) : (
          <button type="button" className="btn btn-scope" onClick={slot.startRecording} disabled={busy}>
            <IconMic /> Record
          </button>
        )}
        {clip && phase !== 'recording' && (
          <button
            type="button"
            className="btn btn-scope"
            onClick={() => slot.togglePlay(!windowed)}
            disabled={phase === 'decoding'}
            aria-pressed={slot.playing}
          >
            {slot.playing ? <IconStop /> : <IconPlay />} {slot.playing ? 'Stop' : windowed ? 'Play window' : 'Play'}
          </button>
        )}
      </div>
      {slot.error && (
        <p className="scope-error" role="alert">
          {slot.error}
        </p>
      )}
    </div>
  )
}

const IconFile = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5M9 14l3-3 3 3M12 11v6" />
  </svg>
)
const IconMic = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </svg>
)
const IconPlay = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" />
  </svg>
)
const IconStop = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
)
