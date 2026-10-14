import type { TicketDraft } from './types'

/** A draft as the server sent it last replaces the earlier copy (same id); oldest first. */
export const upsertTicket = (tickets: TicketDraft[], ticket: TicketDraft): TicketDraft[] =>
  [...tickets.filter((t) => t.id !== ticket.id), ticket].sort((a, b) => a.ts - b.ts)

/** What the Assist panel shows: every ticket except dismissed ones, newest first. */
export const visibleTickets = (tickets: TicketDraft[]): TicketDraft[] =>
  tickets.filter((t) => t.status !== 'dismissed').sort((a, b) => b.ts - a.ts)

/** Tickets filed in Jira during the call, oldest first (the call record). */
export const createdTickets = (tickets: TicketDraft[]): TicketDraft[] =>
  tickets.filter((t) => t.status === 'created' && t.key)

/** The advisor can still edit and file it. */
export const isEditable = (ticket: TicketDraft): boolean => ticket.status === 'ready' || ticket.status === 'failed'
