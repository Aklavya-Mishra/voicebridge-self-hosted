import { domainText, factLabel, visibleHints } from '../lib/assist'
import { visibleTickets } from '../lib/tickets'
import type { NbaHint, NbaStatusEvent, TicketDraft } from '../lib/types'
import { TicketCard } from './TicketCard'

interface Props {
  /** Live hints can run in this call (server has them, not in-room): the Hints switch is shown. */
  hintsAvailable: boolean
  /** Live hints are on for this call (the call type, hint and customer info sections). */
  hintsOn: boolean
  onToggleHints: () => void
  /** Call type detected from the caller's latest line; null until they first speak. */
  detected: string | null
  hints: NbaHint[]
  facts: Record<string, string>
  status: NbaStatusEvent['state']
  reason?: string
  /** Jira is set up on the server: tickets can be asked for on the call. */
  ticketsEnabled: boolean
  tickets: TicketDraft[]
  onCreateTicket: (ticket: TicketDraft, title: string, description: string) => void
  onDismissTicket: (ticket: TicketDraft) => void
}

/** Jira tickets asked for on the call, live hints for the advisor (newest first) and what the customer has told. */
export function AssistPanel({
  hintsAvailable,
  hintsOn,
  onToggleHints,
  detected,
  hints,
  facts,
  status,
  reason,
  ticketsEnabled,
  tickets,
  onCreateTicket,
  onDismissTicket,
}: Props) {
  const shownTickets = visibleTickets(tickets)
  const hintsEnabled = hintsAvailable && hintsOn
  return (
    <aside className="assist card" aria-label="Call assist">
      <div className="assist-head">
        <h2>Assist</h2>
        {hintsAvailable && (
          <button className="hints-toggle" aria-pressed={hintsOn} onClick={onToggleHints}>
            Hints: {hintsOn ? 'On' : 'Off'}
          </button>
        )}
      </div>
      {hintsAvailable && !hintsOn && (
        <p className="muted small">Hints are off for this call. Jira, the summary and the Slack recap still work.</p>
      )}
      {hintsEnabled && <p className="muted small">{domainText(detected)}</p>}
      {hintsEnabled && status === 'thinking' && <p className="muted small">Thinking…</p>}
      {hintsEnabled && status === 'unavailable' && (
        <p className="notice small">AI hints unavailable ({reason}). Rule hints still work.</p>
      )}
      {ticketsEnabled && (
        <>
          <h3>Jira ticket</h3>
          {shownTickets.length === 0 ? (
            <p className="muted small">Say “create a Jira ticket …” on the call to draft one.</p>
          ) : (
            <div className="tickets">
              {shownTickets.map((t) => (
                <TicketCard key={t.id} ticket={t} onCreate={onCreateTicket} onDismiss={onDismissTicket} />
              ))}
            </div>
          )}
        </>
      )}
      {hintsEnabled && (
        <>
          <h3>Next best action</h3>
          <HintList hints={visibleHints(hints)} />
          <h3>Customer info</h3>
          <FactList facts={facts} />
        </>
      )}
    </aside>
  )
}

export function HintList({ hints }: { hints: NbaHint[] }) {
  if (hints.length === 0) return <p className="muted small">Hints appear after the other person speaks.</p>
  return (
    <ol className="hints">
      {hints.map((h, i) => (
        <li key={h.id} className={`hint ${h.priority}${i === 0 ? ' latest' : ''}`}>
          {h.kind === 'rule' && <span className="hint-tag">rule</span>}
          {h.text}
        </li>
      ))}
    </ol>
  )
}

export function FactList({ facts }: { facts: Record<string, string> }) {
  const entries = Object.entries(facts)
  if (entries.length === 0) return <p className="muted small">Nothing yet.</p>
  return (
    <dl className="facts">
      {entries.map(([key, value]) => (
        <div key={key} className="fact">
          <dt>{factLabel(key)}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}
