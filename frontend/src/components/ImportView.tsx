import { useState, type FormEvent } from 'react'
import { importRecording } from '../lib/api'
import { fromUtterance, type Session, type User } from '../lib/types'
import { Chat } from './Chat'

interface Props {
  token: string
  me: User
  onBack: () => void
}

export function ImportView({ token, me, onBack }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [meChannel, setMeChannel] = useState<'left' | 'right'>('left')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [session, setSession] = useState<Session | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!file) return
    setBusy(true)
    setError('')
    try {
      setSession((await importRecording(token, file, meChannel)).session)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (session) {
    const messages = session.transcript.map(fromUtterance)
    return (
      <div className="call">
        <header className="call-header card">
          <div>
            <div className="muted small">Imported recording</div>
            <strong>{file?.name}</strong>
          </div>
          <div className="call-actions">
            <button onClick={() => setSession(null)}>Import another</button>
            <button onClick={onBack}>Back</button>
          </div>
        </header>
        <Chat
          messages={messages}
          myId={me.id}
          originTs={messages[0]?.startTs ?? 0}
          emptyHint="No speech was found in this recording."
        />
      </div>
    )
  }

  return (
    <form className="card narrow" onSubmit={submit}>
      <h1>Import a recording</h1>
      <p className="muted">
        The file must be <strong>stereo with one person per channel</strong>. Mono recordings are rejected because the
        speakers cannot be told apart reliably.
      </p>
      <label htmlFor="file">Audio or video file</label>
      <input
        id="file"
        type="file"
        accept="audio/*,video/*,.wav,.mp3,.m4a,.ogg,.webm,.flac"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      <fieldset className="channel-pick">
        <legend>Which channel is you?</legend>
        <label>
          <input type="radio" checked={meChannel === 'left'} onChange={() => setMeChannel('left')} /> Left
        </label>
        <label>
          <input type="radio" checked={meChannel === 'right'} onChange={() => setMeChannel('right')} /> Right
        </label>
      </fieldset>
      {error && <p className="error">{error}</p>}
      <div className="modal-actions">
        <button type="button" onClick={onBack}>
          Cancel
        </button>
        <button className="primary" disabled={!file || busy}>
          {busy ? 'Transcribing…' : 'Transcribe'}
        </button>
      </div>
    </form>
  )
}
