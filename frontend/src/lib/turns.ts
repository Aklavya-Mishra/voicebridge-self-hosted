import type { ChatMessage } from './types'

/** A silence this long (seconds) from the same speaker starts a new bubble. */
export const TURN_GAP_S = 3

export interface Turn {
  speakerId: string
  speakerName: string
  startTs: number
  parts: ChatMessage[]
}

/** Consecutive lines from the same speaker read as one turn; a speaker change or a long pause starts a new bubble. */
export const toTurns = (messages: ChatMessage[]): Turn[] => {
  const turns: Turn[] = []
  for (const m of messages) {
    const last = turns[turns.length - 1]
    const prev = last?.parts[last.parts.length - 1]
    if (last && prev && last.speakerId === m.speakerId && m.startTs - prev.endTs < TURN_GAP_S) last.parts.push(m)
    else turns.push({ speakerId: m.speakerId, speakerName: m.speakerName, startTs: m.startTs, parts: [m] })
  }
  return turns
}
