import assert from 'node:assert/strict'
import { test } from 'node:test'
import { addHint, domainText, factLabel, hintsNewestFirst, visibleHints } from '../src/lib/assist.ts'
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
  assert.equal(domainText('auto', null), 'Detecting call type…')
  assert.equal(domainText('auto', 'telecom'), 'Telecom · detected')
  assert.equal(domainText('tech_support', null), 'Tech support')
})

test('the live panel shows at most the two newest hints', () => {
  assert.deepEqual(
    visibleHints([hint('c', 3), hint('b', 2), hint('a', 1)]).map((h) => h.id),
    ['c', 'b'],
  )
})
