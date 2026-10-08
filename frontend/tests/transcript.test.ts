import assert from 'node:assert/strict'
import { test } from 'node:test'
import { formatTranscript } from '../src/lib/transcript.ts'
import type { ChatMessage } from '../src/lib/types.ts'

const msg = (id: string, speakerId: string, speakerName: string, text: string, startTs: number, endTs = startTs + 1): ChatMessage => ({
  id,
  speakerId,
  speakerName,
  text,
  startTs,
  endTs,
  final: true,
})

test('uses the name each speaker was recorded with, in speaking order', () => {
  const text = formatTranscript([
    msg('1', 'me', 'Aklavya', 'How can I help you?', 0),
    msg('2', 'me-call', 'Computer', 'I was wondering if I can get help in flight booking.', 2),
    msg('3', 'me', 'Aklavya', 'Sure.', 5),
  ])
  assert.equal(
    text,
    'Aklavya: How can I help you?\nComputer: I was wondering if I can get help in flight booking.\nAklavya: Sure.\n',
  )
})

test('another app user appears under the name they entered', () => {
  const text = formatTranscript([msg('1', 'me', 'Aklavya', 'Hi Ben.', 0), msg('2', 'ben', 'Ben', 'Hello!', 2)])
  assert.equal(text, 'Aklavya: Hi Ben.\nBen: Hello!\n')
})

test('lines of the same speaker stay separate, one per transcript line', () => {
  const text = formatTranscript([msg('1', 'me', 'Aklavya', 'Hello,', 0, 1), msg('2', 'me', 'Aklavya', 'how are you?', 1.5, 2)])
  assert.equal(text, 'Aklavya: Hello,\nAklavya: how are you?\n')
})

test('empty call gives an empty transcript', () => {
  assert.equal(formatTranscript([]), '')
})
