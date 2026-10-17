import type { ChatMessage } from './types'

/** In-room speaker ids look like "<user id>-room2". */
export const isRoomSpeaker = (speakerId: string): boolean => /-room\d+$/.test(speakerId)

/** Voices in my call audio look like "<my id>-call", "<my id>-call2"... */
export const isCallSpeaker = (speakerId: string, myId: string): boolean =>
  speakerId.startsWith(`${myId}-call`) && /^\d*$/.test(speakerId.slice(myId.length + 5))

/** The voices heard in my call audio, in the order they were first heard. */
export const callSpeakerIds = (messages: ChatMessage[], myId: string): string[] => [
  ...new Set(messages.filter((m) => isCallSpeaker(m.speakerId, myId)).map((m) => m.speakerId)),
]

/** A call voice's current name: renamed in the call, else the name on its latest line, else "Computer n". */
export const callSpeakerName = (id: string, messages: ChatMessage[], renamed: Record<string, string>): string => {
  const n = id.split('-call').pop()
  return (
    renamed[id] ??
    [...messages].reverse().find((m) => m.speakerId === id)?.speakerName ??
    (n ? `Computer ${n}` : 'Computer')
  )
}

/** The meeting's speakers in the order they were first heard. */
export const roomSpeakerIds = (messages: ChatMessage[]): string[] => [
  ...new Set(messages.filter((m) => isRoomSpeaker(m.speakerId)).map((m) => m.speakerId)),
]

/** A speaker's current name: renamed in the meeting, else the name on their latest line, else "Speaker n". */
export const roomSpeakerName = (id: string, messages: ChatMessage[], renamed: Record<string, string>): string =>
  renamed[id] ??
  [...messages].reverse().find((m) => m.speakerId === id)?.speakerName ??
  `Speaker ${id.split('-room').pop()}`
