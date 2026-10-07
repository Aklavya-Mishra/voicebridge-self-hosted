export interface User {
  id: string
  name: string
}

export interface Utterance {
  id: string
  speaker_id: string
  speaker_name: string
  text: string
  start_ts: number
  end_ts: number
}

export interface Session {
  id: string
  code: string
  kind: 'live' | 'import'
  created_by: string
  created_at: number
  participants: User[]
  transcript: Utterance[]
  summary: CallSummary | null
  /** Names given to speakers who are not participants, e.g. "<user id>-call" renamed from "Computer". */
  speaker_names?: Record<string, string>
}

/** End-of-call analysis, made by the backend once a call stops. */
export type CallSummary =
  | { status: 'pending' }
  | { status: 'unavailable' }
  | {
      status: 'ready'
      provider: string
      summary: string
      purpose: string
      action_items: string[]
      outcome: string
      sentiment: 'positive' | 'neutral' | 'negative'
      // quality: number  // disabled in the backend (app/analysis.py)
    }

/** Server `transcript` event. A final with empty text means "drop this bubble". */
export interface TranscriptEvent {
  code: string
  utterance_id: string
  speaker: User
  text: string
  start_ts: number
  end_ts: number
  final: boolean
}

/** Server `stt_status` event: a speaker's transcription engine changed (Deepgram -> local fallback). */
export interface SttStatusEvent {
  code: string
  speaker: User
  engine: 'deepgram' | 'local'
  reason: string
}

/** A saved call in the "Past calls" list. */
export interface SessionSummary {
  code: string
  kind: 'live' | 'import'
  created_at: number
  participants: User[]
  lines: number
  preview: string
  summary: { status: CallSummary['status']; sentiment?: string } | null
}

export interface ChatMessage {
  id: string
  speakerId: string
  speakerName: string
  text: string
  startTs: number
  endTs: number
  final: boolean
}

export const fromUtterance = (u: Utterance): ChatMessage => ({
  id: u.id,
  speakerId: u.speaker_id,
  speakerName: u.speaker_name,
  text: u.text,
  startTs: u.start_ts,
  endTs: u.end_ts,
  final: true,
})
