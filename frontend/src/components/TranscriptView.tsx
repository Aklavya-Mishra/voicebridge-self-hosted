import { useEffect, useState } from 'react'
import { getSession } from '../lib/api'
import { formatTranscript } from '../lib/transcript'
import { fromUtterance, type Session } from '../lib/types'

interface Props {
  code: string
  token: string
  onBack: () => void
}

/** A saved call as its transcript only: "Name: text" lines, and Back. */
export function TranscriptView({ code, token, onBack }: Props) {
  const [session, setSession] = useState<Session | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getSession(token, code)
      .then((r) => setSession(r.session))
      .catch((err: Error) => setError(err.message))
  }, [code, token])

  const lines = session
    ? formatTranscript(session.transcript.map(fromUtterance))
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
      {session && lines.length === 0 && <p className="muted">This call has no transcript.</p>}
      {lines.map((line, i) => (
        <p key={i} className="transcript-line">
          <strong>{line.name}:</strong> {line.text}
        </p>
      ))}
    </div>
  )
}
