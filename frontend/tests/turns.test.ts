import assert from 'node:assert/strict'
import { test } from 'node:test'
import { toTurns } from '../src/lib/turns.ts'
import type { ChatMessage } from '../src/lib/types.ts'

const msg = (id: string, speakerId: string, startTs: number, endTs: number): ChatMessage => ({
  id,
  speakerId,
  speakerName: speakerId,
  text: id,
  startTs,
  endTs,
  final: true,
})

test('same speaker with short pauses stays in one bubble', () => {
  const turns = toTurns([msg('a1', 'a', 0, 2), msg('a2', 'a', 4.9, 6)])
  assert.deepEqual(
    turns.map((t) => t.parts.map((p) => p.id)),
    [['a1', 'a2']],
  )
})

test('same speaker pausing 3 s or more starts a new bubble', () => {
  const turns = toTurns([msg('a1', 'a', 0, 2), msg('a2', 'a', 5, 6), msg('a3', 'a', 10, 11)])
  assert.deepEqual(
    turns.map((t) => t.parts.map((p) => p.id)),
    [['a1'], ['a2'], ['a3']],
  )
})

test('speaker change starts a new bubble', () => {
  const turns = toTurns([msg('a1', 'a', 0, 2), msg('b1', 'b', 2.5, 3)])
  assert.deepEqual(
    turns.map((t) => t.speakerId),
    ['a', 'b'],
  )
})

test('with no gap limit, a speaker keeps one turn until someone else speaks', () => {
  const turns = toTurns([msg('a1', 'a', 0, 2), msg('a2', 'a', 30, 31), msg('b1', 'b', 32, 33), msg('a3', 'a', 60, 61)], Infinity)
  assert.deepEqual(
    turns.map((t) => t.parts.map((p) => p.id)),
    [['a1', 'a2'], ['b1'], ['a3']],
  )
})
