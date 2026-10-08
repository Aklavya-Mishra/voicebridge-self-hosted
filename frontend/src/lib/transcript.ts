import type { ChatMessage } from './types'

/**
 * The call as plain text, one line per transcript line, labelled with the name each speaker was recorded with:
 * the name they entered for the call, or "Computer" for audio captured from the computer.
 */
export const formatTranscript = (messages: ChatMessage[]): string =>
  messages.map((m) => `${m.speakerName}: ${m.text}\n`).join('')
