import { toTurns } from './turns.ts' // extension: also loaded by `node --test`
import type { ChatMessage } from './types'

/** Label for everyone who is not me: the call side captured from the computer's audio. */
export const OTHER_LABEL = 'Computer'

/** The call as plain text, one line per bubble: "Aklavya: ..." / "Computer: ...". */
export const formatTranscript = (messages: ChatMessage[], myId: string, myName: string): string =>
  toTurns(messages)
    .map((turn) => `${turn.speakerId === myId ? myName : OTHER_LABEL}: ${turn.parts.map((m) => m.text).join(' ')}\n`)
    .join('')
