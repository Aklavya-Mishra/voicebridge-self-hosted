import { useEffect, useState } from 'react'
import { listSessions } from '../lib/api'
import type { SessionSummary } from '../lib/types'

interface Props {
  token: string
  onOpen: (code: string) => void
}

/** Saved calls, newest first; opening one shows its transcript (and allows the Transcript download). */
export function PastCalls({ token, onOpen }: Props) {
  const [calls, setCalls] = useState<SessionSummary[] | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    listSessions(token)
      .then((r) => setCalls(r.sessions))
      .catch((err: Error) => setError(err.message))
  }, [token])

  return (
    <section className="card past-calls">
      <h2>Past calls</h2>
      {error && <p className="error">{error}</p>}
      {calls?.length === 0 && <p className="muted">No calls yet. Calls are saved automatically.</p>}
      <ul>
        {calls?.map((call) => (
          <li key={call.code}>
            <button className="past-call" onClick={() => onOpen(call.code)}>
              <span className="past-call-when">{new Date(call.created_at * 1000).toLocaleString()}</span>
              <span className="past-call-preview">{call.preview || 'No transcript'}</span>
              <span className="muted small">
                {call.kind === 'import' ? 'Import' : call.code} · {call.lines} line{call.lines === 1 ? '' : 's'}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
