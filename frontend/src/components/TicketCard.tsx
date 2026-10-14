import { useState } from 'react'
import { isEditable } from '../lib/tickets'
import type { TicketDraft } from '../lib/types'

interface Props {
  ticket: TicketDraft
  onCreate: (ticket: TicketDraft, title: string, description: string) => void
  onDismiss: (ticket: TicketDraft) => void
}

/** One Jira ticket drafted from the call: the advisor checks and edits it, then files it (human in the loop). */
export function TicketCard({ ticket, onCreate, onDismiss }: Props) {
  if (ticket.status === 'drafting') {
    return (
      <div className="ticket drafting" aria-live="polite">
        <p className="small">Drafting a Jira ticket…</p>
        <p className="muted small ticket-trigger">
          {ticket.requested_by}: “{ticket.trigger}”
        </p>
      </div>
    )
  }
  if (ticket.status === 'created') {
    return (
      <div className="ticket created">
        <p className="small">
          Created{' '}
          <a href={ticket.url} target="_blank" rel="noreferrer">
            {ticket.key}
          </a>
        </p>
        <p className="small ticket-title">{ticket.title}</p>
      </div>
    )
  }
  // A new draft id remounts the form, so the fields start from the model's text.
  return <TicketForm key={ticket.id} ticket={ticket} onCreate={onCreate} onDismiss={onDismiss} />
}

function TicketForm({ ticket, onCreate, onDismiss }: Props) {
  const [title, setTitle] = useState(ticket.title)
  const [description, setDescription] = useState(ticket.description)
  const editable = isEditable(ticket)
  const titleId = `ticket-title-${ticket.id}`
  const descriptionId = `ticket-description-${ticket.id}`

  return (
    <form
      className="ticket"
      onSubmit={(e) => {
        e.preventDefault()
        onCreate(ticket, title, description)
      }}
    >
      <p className="muted small ticket-trigger">
        {ticket.requested_by}: “{ticket.trigger}”
      </p>
      <label htmlFor={titleId}>Title</label>
      <input
        id={titleId}
        value={title}
        maxLength={255}
        disabled={!editable}
        onChange={(e) => setTitle(e.target.value)}
      />
      <label htmlFor={descriptionId}>Description</label>
      <textarea
        id={descriptionId}
        rows={3}
        value={description}
        disabled={!editable}
        onChange={(e) => setDescription(e.target.value)}
      />
      {ticket.status === 'failed' && <p className="error small">{ticket.error}</p>}
      <div className="ticket-actions">
        <button type="submit" className="primary" disabled={!editable || !title.trim()}>
          {ticket.status === 'creating' ? 'Creating…' : ticket.status === 'failed' ? 'Try again' : 'Create ticket'}
        </button>
        <button type="button" onClick={() => onDismiss(ticket)} disabled={!editable}>
          Dismiss
        </button>
      </div>
    </form>
  )
}
