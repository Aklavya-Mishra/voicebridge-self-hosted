import { useEffect, useState } from 'react'
import { getSession } from '../lib/api'
import { formatTranscript } from '../lib/transcript'
import { fromUtterance, type CallSummary, type Session } from '../lib/types'

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
  const pending = session?.summary?.status === 'pending'

  useEffect(() => {
    getSession(token, code)
      .then((r) => setSession(r.session))
      .catch((err: Error) => setError(err.message))
  }, [code, token])

  // While the summary is being made (right after a call), check back until it is ready.
  useEffect(() => {
    if (!pending) return
    const timer = setInterval(() => {
      getSession(token, code)
        .then((r) => setSession(r.session))
        .catch(() => undefined)
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [pending, code, token])

  const lines = session
    ? formatTranscript(session.transcript.map(fromUtterance), Infinity) // one paragraph per speaker turn
        .split('\n')
        .filter(Boolean)
        .map((line) => {
          const i = line.indexOf(': ')
          return { name: line.slice(0, i), text: line.slice(i + 2) }
        })
    : []

  return (
    <div className="card transcript-view">
      <div className="transcript-head">
        <button onClick={onBack}>← Back</button>
        {session && <span className="muted small">{new Date(session.created_at * 1000).toLocaleString()}</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {session && lines.length > 0 && <SummaryBlock summary={session.summary} />}
      {session && lines.length === 0 && <p className="muted">This call has no transcript.</p>}
      {lines.map((line, i) => (
        <p key={i} className="transcript-line">
          <strong>{line.name}:</strong> {line.text}
        </p>
      ))}
    </div>
  )
}

function SummaryBlock({ summary }: { summary: CallSummary | null }) {
  if (!summary) return null
  if (summary.status === 'pending') return <p className="muted summary-box">Preparing summary…</p>
  if (summary.status === 'unavailable') return <p className="muted summary-box">Summary unavailable.</p>
  return (
    <section className="summary-box" aria-label="Call summary">
      <div className="summary-tags">
        <span className={`pill sentiment-${summary.sentiment}`}>{summary.sentiment}</span>
        {/* Quality score is disabled in the backend (app/analysis.py); restore with it:
        <span className="pill">Quality {summary.quality}/5</span> */}
      </div>
      <p>{summary.summary}</p>
      <dl>
        <dt>Purpose</dt>
        <dd>{summary.purpose}</dd>
        <dt>Outcome</dt>
        <dd>{summary.outcome}</dd>
        {summary.action_items.length > 0 && (
          <>
            <dt>Action items</dt>
            <dd>
              <ul>
                {summary.action_items.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </dd>
          </>
        )}
      </dl>
    </section>
  )
}
