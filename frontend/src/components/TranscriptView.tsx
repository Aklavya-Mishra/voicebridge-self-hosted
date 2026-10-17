import { useEffect, useState } from 'react'
import { getSession, health, sendRecap } from '../lib/api'
import { savedFacts } from '../lib/assist'
import { createdTickets } from '../lib/tickets'
import { fromUtterance, type CallSummary, type MeetingMinutes, type Session, type SlackRecap } from '../lib/types'
import { FactList } from './AssistPanel'

interface Props {
  code: string
  token: string
  onBack: () => void
}

const POLL_MS = 3000

/** A saved call: its summary, then the transcript as "Name: text" lines, and Back. */
export function TranscriptView({ code, token, onBack }: Props) {
  const [session, setSession] = useState<Session | null>(null)
  const [error, setError] = useState('')
  const [slackOn, setSlackOn] = useState(false)
  // Right after a call: the summary is being made, or it is ready and its Slack recap is on its way.
  const pending =
    session?.summary?.status === 'pending' ||
    (slackOn && session?.kind === 'live' && session.summary?.status === 'ready' && !session.recap)

  useEffect(() => {
    health()
      .then((h) => setSlackOn(Boolean(h.slack_recap)))
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    getSession(token, code)
      .then((r) => setSession(r.session))
      .catch((err: Error) => setError(err.message))
  }, [code, token])

  // Until the summary (and its recap) is ready, check back.
  useEffect(() => {
    if (!pending) return
    const timer = setInterval(() => {
      getSession(token, code)
        .then((r) => setSession(r.session))
        .catch(() => undefined)
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [pending, code, token])

  const lines = session ? session.transcript.map(fromUtterance) : []

  return (
    <div className="card transcript-view">
      <div className="transcript-head">
        <button onClick={onBack}>← Back</button>
        {session && <span className="muted small">{new Date(session.created_at * 1000).toLocaleString()}</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {session && lines.length > 0 && <SummaryBlock summary={session.summary} />}
      {session && slackOn && session.kind === 'live' && session.summary?.status === 'ready' && (
        <RecapLine
          recap={session.recap ?? null}
          onRetry={() =>
            sendRecap(token, code)
              .then((r) => setSession({ ...session, recap: r.recap }))
              .catch((err: Error) => setSession({ ...session, recap: { status: 'failed', error: err.message } }))
          }
        />
      )}
      {session && session.nba && Object.keys(savedFacts(session.nba)).length > 0 && (
        <section className="summary-box" aria-label="Customer info">
          {/* Hints during the call are hidden for now; restore these two lines to show them again.
          <h3>Hints during the call</h3>
          <HintList hints={hintsNewestFirst(session.nba.hints)} />
          */}
          <h3>Customer info</h3>
          <FactList facts={savedFacts(session.nba)} />
        </section>
      )}
      {session && createdTickets(session.tickets ?? []).length > 0 && (
        <section className="summary-box" aria-label="Jira tickets">
          <h3>Jira tickets</h3>
          <ul className="ticket-list">
            {createdTickets(session.tickets ?? []).map((t) => (
              <li key={t.id}>
                <a href={t.url} target="_blank" rel="noreferrer">
                  {t.key}
                </a>{' '}
                {t.title}
              </li>
            ))}
          </ul>
        </section>
      )}
      {session && lines.length === 0 && <p className="muted">This call has no transcript.</p>}
      {lines.map((line) => (
        <p key={line.id} className="transcript-line">
          <strong>{line.speakerName}:</strong> {line.text}
        </p>
      ))}
    </div>
  )
}

function RecapLine({ recap, onRetry }: { recap: SlackRecap | null; onRetry: () => Promise<void> }) {
  const [busy, setBusy] = useState(false)
  if (!recap) return <p className="muted small recap-line">Sending recap to Slack…</p>
  const channel = recap.name ? `#${recap.name}` : 'Slack'
  if (recap.status === 'sent')
    return (
      <p className="small recap-line" role="status">
        ✓ Recap sent to {channel}
        {recap.permalink && (
          <>
            {' · '}
            <a href={recap.permalink} target="_blank" rel="noreferrer">
              Open in Slack
            </a>
          </>
        )}
      </p>
    )
  return (
    <p className="small recap-line error" role="alert">
      Recap not sent to Slack: {recap.error || 'unknown error'}{' '}
      <button
        disabled={busy}
        onClick={() => {
          setBusy(true)
          void onRetry().finally(() => setBusy(false))
        }}
      >
        {busy ? 'Sending…' : 'Retry'}
      </button>
    </p>
  )
}

function SummaryBlock({ summary }: { summary: CallSummary | null }) {
  if (!summary) return null
  if (summary.status === 'pending') return <p className="muted summary-box">Preparing summary…</p>
  if (summary.status === 'unavailable') return <p className="muted summary-box">Summary unavailable.</p>
  if (summary.kind === 'minutes') return <MinutesBlock minutes={summary} />
  return (
    <section className="summary-box" aria-label="Call summary">
      <div className="summary-tags">
        {summary.sentiment && (
          <span className={`pill sentiment-${summary.sentiment}`}>Sentiment: {summary.sentiment}</span>
        )}
      </div>
      <p>{summary.summary}</p>
      <dl>
        <dt>Purpose</dt>
        <dd>{summary.purpose}</dd>
        <dt>Outcome</dt>
        <dd>{summary.outcome}</dd>
        <dt>Next steps</dt>
        <dd>
          {summary.action_items.length > 0 ? (
            <ul>
              {summary.action_items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          ) : (
            'None agreed'
          )}
        </dd>
      </dl>
    </section>
  )
}

function ItemList({ items, empty }: { items: string[]; empty: string }) {
  if (items.length === 0) return <>{empty}</>
  return (
    <ul>
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ul>
  )
}

/** Minutes of an in-room meeting. */
function MinutesBlock({ minutes }: { minutes: MeetingMinutes }) {
  return (
    <section className="summary-box" aria-label="Minutes of the meeting">
      <h3>Minutes of the meeting</h3>
      <p>{minutes.summary}</p>
      <dl>
        <dt>Attendees</dt>
        <dd>{minutes.attendees.join(', ')}</dd>
        <dt>Discussion</dt>
        <dd>
          <ItemList items={minutes.discussion_points} empty="Nothing recorded" />
        </dd>
        <dt>Decisions</dt>
        <dd>
          <ItemList items={minutes.decisions} empty="None" />
        </dd>
        <dt>Next steps</dt>
        <dd>
          <ItemList items={minutes.action_items} empty="None agreed" />
        </dd>
        <dt>Open issues</dt>
        <dd>
          <ItemList items={minutes.open_issues} empty="None" />
        </dd>
        {minutes.done.length > 0 && (
          <>
            <dt>Done in the meeting</dt>
            <dd>
              <ItemList items={minutes.done} empty="" />
            </dd>
          </>
        )}
      </dl>
    </section>
  )
}
