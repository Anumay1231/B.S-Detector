import { useRef, useState } from 'react'
import { WINDOW_SECONDS, formatSeconds } from '../lib/audio'
import { RECORD_LIMIT, type ClipSlot } from '../lib/useClip'
import { Scope } from './Scope'
import './VoiceInput.css'

interface Props {
  slot: ClipSlot
  title: string
  helper: string
  placeholder: string
  /** Show and let people move the 4.04 s window the deepfake check hears. */
  windowed: boolean
  locked: boolean
  /** Optional slots get a remove button once filled. */
  optional?: boolean
}

export function VoiceInput({ slot, title, helper, placeholder, windowed, locked, optional }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const { clip, phase } = slot
  const busy = slot.busy || locked
  const recording = phase === 'recording'
  const end = clip ? Math.min(clip.duration, slot.windowStart + WINDOW_SECONDS) : 0

  let line: React.ReactNode = <span className="composer-placeholder">{placeholder}</span>
  if (recording)
    line = (
      <span className="composer-recording">
        Recording <span className="num">{slot.recordSeconds.toFixed(1)}</span> of {RECORD_LIMIT} s
      </span>
    )
  else if (phase === 'decoding') line = <span className="composer-placeholder">Reading audio…</span>
  else if (clip)
    line = (
      <span className="composer-file">
        <span className="composer-name" title={clip.name}>
          {clip.name}
        </span>
        <span className="tag num">{formatSeconds(clip.duration)}</span>
      </span>
    )

  return (
    <div
      className={`voice${dragOver ? ' is-dragover' : ''}`}
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
      <div className="voice-head">
        <h3>{title}</h3>
        <p>{helper}</p>
      </div>

      <div className="composer">
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm"
          className="visually-hidden"
          tabIndex={-1}
          aria-label={`${title}: audio file`}
          onChange={(e) => {
            slot.loadFiles(e.target.files)
            e.target.value = ''
          }}
        />
        <button type="button" className="icon-btn" onClick={() => fileRef.current?.click()} disabled={busy} aria-label={`${title}: choose a file`} title="Choose a file">
          <IconUpload />
        </button>
        {recording ? (
          <button type="button" className="icon-btn is-recording" onClick={slot.stopRecording} aria-label="Stop recording" title="Stop recording">
            <IconStop />
          </button>
        ) : (
          <button type="button" className="icon-btn" onClick={slot.startRecording} disabled={busy} aria-label={`${title}: record from the microphone`} title="Record">
            <IconMic />
          </button>
        )}
        <div className="composer-line" aria-live="polite">
          {line}
        </div>
        {clip && !recording && (
          <button
            type="button"
            className="icon-btn icon-btn-dark"
            onClick={() => slot.togglePlay(!windowed)}
            disabled={phase === 'decoding'}
            aria-pressed={slot.playing}
            aria-label={slot.playing ? 'Stop playback' : windowed ? 'Play the analysis window' : 'Play the clip'}
            title={slot.playing ? 'Stop' : 'Play'}
          >
            {slot.playing ? <IconStop /> : <IconPlay />}
          </button>
        )}
        {optional && clip && !recording && (
          <button type="button" className="icon-btn" onClick={slot.clear} disabled={busy} aria-label={`Remove the ${title.toLowerCase()}`} title="Remove">
            <IconClose />
          </button>
        )}
      </div>

      {(clip || recording) && (
        <div className="voice-scope">
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
          {clip && !recording && (
            <p className="voice-window tag">
              {windowed ? (
                <>
                  The deepfake check hears <span className="num">{formatSeconds(slot.windowStart)}</span> to{' '}
                  <span className="num">{formatSeconds(end)}</span>
                  {clip.duration > WINDOW_SECONDS && '. Drag the bracket, or use the arrow keys, to choose another part.'}
                </>
              ) : (
                'The speaker check uses the whole clip.'
              )}
            </p>
          )}
        </div>
      )}

      {slot.error && (
        <p className="voice-error" role="alert">
          {slot.error}
        </p>
      )}
    </div>
  )
}

const stroke = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round' } as const

const IconUpload = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" {...stroke} aria-hidden="true">
    <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />
  </svg>
)
const IconMic = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" {...stroke} aria-hidden="true">
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </svg>
)
const IconPlay = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" />
  </svg>
)
const IconStop = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <rect x="5" y="5" width="14" height="14" rx="2.5" />
  </svg>
)
const IconClose = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" {...stroke} aria-hidden="true">
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
)
