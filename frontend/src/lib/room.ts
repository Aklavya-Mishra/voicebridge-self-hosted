import type { ChatMessage } from './types'

/** In-room speaker ids look like "<user id>-room2". */
export const isRoomSpeaker = (speakerId: string): boolean => /-room\d+$/.test(speakerId)

/** The meeting's speakers in the order they were first heard. */
export const roomSpeakerIds = (messages: ChatMessage[]): string[] => [
  ...new Set(messages.filter((m) => isRoomSpeaker(m.speakerId)).map((m) => m.speakerId)),
]

/** A speaker's current name: renamed in the meeting, else the name on their latest line, else "Speaker n". */
export const roomSpeakerName = (id: string, messages: ChatMessage[], renamed: Record<string, string>): string =>
  renamed[id] ??
  [...messages].reverse().find((m) => m.speakerId === id)?.speakerName ??
  `Speaker ${id.split('-room').pop()}`
