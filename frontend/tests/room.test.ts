import assert from 'node:assert/strict'
import { test } from 'node:test'
import { isRoomSpeaker, roomSpeakerIds, roomSpeakerName } from '../src/lib/room.ts'
import type { ChatMessage } from '../src/lib/types.ts'

const msg = (id: string, speakerId: string, speakerName: string, startTs: number): ChatMessage => ({
  id,
  speakerId,
  speakerName,
  text: 'x',
  startTs,
  endTs: startTs + 1,
  final: true,
})

test('room speakers are listed in the order they were first heard', () => {
  const messages = [msg('a', 'u-room2', 'Speaker 2', 1), msg('b', 'u-room1', 'Speaker 1', 2), msg('c', 'u-room2', 'Speaker 2', 3)]
  assert.deepEqual(roomSpeakerIds(messages), ['u-room2', 'u-room1'])
})

test('only in-room speaker ids count', () => {
  assert.equal(isRoomSpeaker('abc-room12'), true)
  assert.equal(isRoomSpeaker('abc-call'), false)
  assert.equal(isRoomSpeaker('abc'), false)
})

test('a speaker is shown by their new name, else their latest line, else their number', () => {
  const messages = [msg('a', 'u-room1', 'Speaker 1', 1), msg('b', 'u-room1', 'Vikram', 2)]
  assert.equal(roomSpeakerName('u-room1', messages, { 'u-room1': 'Vikram S' }), 'Vikram S')
  assert.equal(roomSpeakerName('u-room1', messages, {}), 'Vikram')
  assert.equal(roomSpeakerName('u-room3', messages, {}), 'Speaker 3')
})
