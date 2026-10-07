import { toTurns } from './turns.ts' // extension: also loaded by `node --test`
import type { ChatMessage } from './types'

/**
 * The call as plain text, one line per bubble, labelled with the name each speaker was recorded with:
 * the name they entered for the call, or "Computer" for audio captured from the computer.
 */
export const formatTranscript = (messages: ChatMessage[]): string =>
  toTurns(messages)
    .map((turn) => `${turn.speakerName}: ${turn.parts.map((m) => m.text).join(' ')}\n`)
    .join('')
