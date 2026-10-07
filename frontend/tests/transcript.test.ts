import assert from 'node:assert/strict'
import { test } from 'node:test'
import { formatTranscript } from '../src/lib/transcript.ts'
import type { ChatMessage } from '../src/lib/types.ts'

const msg = (id: string, speakerId: string, text: string, startTs: number, endTs = startTs + 1): ChatMessage => ({
  id,
  speakerId,
  speakerName: speakerId,
  text,
  startTs,
  endTs,
  final: true,
})

test('labels my lines with my name and everyone else as Computer, in speaking order', () => {
  const text = formatTranscript(
    [
      msg('1', 'me', 'How can I help you?', 0),
      msg('2', 'me-call', 'I was wondering if I can get help in flight booking.', 2),
      msg('3', 'me', 'Sure.', 5),
    ],
    'me',
    'Aklavya',
  )
  assert.equal(
    text,
    'Aklavya: How can I help you?\nComputer: I was wondering if I can get help in flight booking.\nAklavya: Sure.\n',
  )
})

test('one line per bubble: lines of the same turn are joined', () => {
  const text = formatTranscript([msg('1', 'me', 'Hello,', 0, 1), msg('2', 'me', 'how are you?', 1.5, 2)], 'me', 'Aklavya')
  assert.equal(text, 'Aklavya: Hello, how are you?\n')
})

test('a long pause from the same speaker starts a new line, as on screen', () => {
  const text = formatTranscript([msg('1', 'me', 'One.', 0, 1), msg('2', 'me', 'Two.', 5, 6)], 'me', 'Aklavya')
  assert.equal(text, 'Aklavya: One.\nAklavya: Two.\n')
})

test('empty call gives an empty transcript', () => {
  assert.equal(formatTranscript([], 'me', 'Aklavya'), '')
})
