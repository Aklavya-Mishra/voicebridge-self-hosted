import assert from 'node:assert/strict'
import { test } from 'node:test'
import { addHint, domainText, factLabel, hintsNewestFirst, savedFacts, visibleHints } from '../src/lib/assist.ts'
import type { NbaHint } from '../src/lib/types.ts'

const hint = (id: string, ts: number, text = `hint ${id}`): NbaHint => ({
  id,
  text,
  kind: 'ai',
  intent: null,
  priority: 'suggestion',
  ts,
})

test('a new hint goes on top', () => {
  assert.deepEqual(
    addHint([hint('a', 1)], hint('b', 2)).map((h) => h.id),
    ['b', 'a'],
  )
})

test('the same hint arriving twice is shown once', () => {
  assert.deepEqual(
    addHint([hint('b', 2), hint('a', 1)], hint('b', 2, 'updated')).map((h) => h.text),
    ['updated', 'hint a'],
  )
})

test('saved hints are shown newest first', () => {
  assert.deepEqual(
    hintsNewestFirst([hint('a', 1), hint('c', 3), hint('b', 2)]).map((h) => h.id),
    ['c', 'b', 'a'],
  )
})

test('fact keys read as words', () => {
  assert.equal(factLabel('current_plan'), 'Current plan')
  assert.equal(factLabel('os'), 'OS')
})

test('call type text', () => {
  assert.equal(domainText(null), 'Detecting call type…')
  assert.equal(domainText('telecom'), 'Call type: Telecom')
  assert.equal(domainText('general'), 'Call type: General')
  assert.equal(domainText('tech_support'), 'Call type: Tech support')
})

test('the live panel shows at most the two newest hints', () => {
  assert.deepEqual(
    visibleHints([hint('c', 3), hint('b', 2), hint('a', 1)]).map((h) => h.id),
    ['c', 'b'],
  )
})

test('the call record shows every value a fact had during the call', () => {
  assert.deepEqual(
    savedFacts({
      hints: [],
      detected: null,
      facts: { budget: '$10/hour', timeline: 'six months' },
      fact_log: { budget: ['$10-$20/hour by role', '$10/hour'] },
    }),
    { budget: '$10-$20/hour by role; $10/hour', timeline: 'six months' },
  )
})
