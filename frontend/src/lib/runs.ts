import type { Analysis, Verification } from './api'

interface RunBase {
  id: number
  at: Date
  /** the clip stage 1 listened to (in verify mode, the voice to check) */
  clipName: string
  clipDuration: number
  windowStart: number
  analysis: Analysis
}

export interface DetectRun extends RunBase {
  mode: 'detect'
}

export interface VerifyRun extends RunBase {
  mode: 'verify'
  referenceName: string
  interlocked: boolean
  verification: Verification | null
}

export type Run = DetectRun | VerifyRun
