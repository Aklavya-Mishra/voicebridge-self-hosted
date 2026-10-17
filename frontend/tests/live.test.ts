import assert from 'node:assert/strict'
import { test } from 'node:test'
import { keepShown, revealDelayMs, sharedWords } from '../src/lib/live.ts'
import { callSpeakerIds, callSpeakerName, isCallSpeaker } from '../src/lib/room.ts'
import type { ChatMessage } from '../src/lib/types.ts'

const w = (s: string) => s.split(' ')

test('a longer version keeps every word already shown', () => {
  assert.equal(keepShown(3, w('morning everyone balram'), w('morning everyone balram priya sara')), 3)
})

test('a corrected word is shown again from where the text changed', () => {
  assert.equal(sharedWords(w('morning aquavia we have'), w('morning aklavya we have the')), 1)
  assert.equal(keepShown(4, w('morning aquavia we have'), w('morning aklavya we have the')), 1)
})

test('words catch up within about a second, never too fast or too slow', () => {
  assert.equal(revealDelayMs(1), 160)
  assert.equal(revealDelayMs(9), 100)
  assert.equal(revealDelayMs(100), 40)
})

const msg = (speakerId: string, speakerName: string, startTs: number): ChatMessage =>
  ({ id: `${speakerId}${startTs}`, speakerId, speakerName, text: 'hi', startTs, final: true }) as ChatMessage

test('call voices are mine only, in the order heard, named Computer then Computer n', () => {
  assert.ok(isCallSpeaker('u1-call', 'u1') && isCallSpeaker('u1-call3', 'u1'))
  assert.ok(!isCallSpeaker('u1-room2', 'u1') && !isCallSpeaker('u12-call', 'u1') && !isCallSpeaker('u1-callx', 'u1'))
  const messages = [msg('u1', 'Aklavya', 1), msg('u1-call2', 'Computer 2', 2), msg('u1-call', 'Computer', 3)]
  assert.deepEqual(callSpeakerIds(messages, 'u1'), ['u1-call2', 'u1-call'])
  assert.equal(callSpeakerName('u1-call3', [], {}), 'Computer 3')
  assert.equal(callSpeakerName('u1-call2', messages, { 'u1-call2': 'Balram' }), 'Balram')
})
