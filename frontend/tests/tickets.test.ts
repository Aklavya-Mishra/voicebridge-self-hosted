import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createdTickets, isEditable, upsertTicket, visibleTickets } from '../src/lib/tickets.ts'
import type { TicketDraft } from '../src/lib/types.ts'

const draft = (id: string, ts: number, status: TicketDraft['status'] = 'ready', key?: string): TicketDraft => ({
  id,
  status,
  title: `title ${id}`,
  description: '',
  requested_by: 'Computer',
  trigger: 'create a ticket',
  ts,
  key,
})

test('a newer copy of a draft replaces the old one', () => {
  const tickets = upsertTicket([draft('a', 1, 'drafting'), draft('b', 2)], draft('a', 1, 'ready'))
  assert.deepEqual(
    tickets.map((t) => [t.id, t.status]),
    [
      ['a', 'ready'],
      ['b', 'ready'],
    ],
  )
})

test('the panel shows newest first and hides dismissed drafts', () => {
  const shown = visibleTickets([draft('a', 1, 'created', 'SCRUM-5'), draft('b', 2, 'dismissed'), draft('c', 3)])
  assert.deepEqual(
    shown.map((t) => t.id),
    ['c', 'a'],
  )
})

test('the call record lists only created tickets', () => {
  const tickets = [draft('a', 1, 'created', 'SCRUM-5'), draft('b', 2, 'failed'), draft('c', 3, 'created', 'SCRUM-6')]
  assert.deepEqual(
    createdTickets(tickets).map((t) => t.key),
    ['SCRUM-5', 'SCRUM-6'],
  )
})

test('only ready and failed drafts can be edited', () => {
  assert.deepEqual(
    (['drafting', 'ready', 'creating', 'created', 'failed', 'dismissed'] as const).map((s) => isEditable(draft('a', 1, s))),
    [false, true, false, false, true, false],
  )
})
