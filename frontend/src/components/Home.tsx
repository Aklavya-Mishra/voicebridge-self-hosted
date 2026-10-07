import { useState, type FormEvent } from 'react'
import { createSession } from '../lib/api'
import { loadCallName, saveCallName } from '../lib/storage'
import { PastCalls } from './PastCalls'

interface Props {
  token: string
  /** Pre-fills "Your name in this call" when no name was entered before. */
  defaultName: string
  /** Opens a live call; `callName` is recorded on this user's lines. */
  onOpenCall: (code: string, callName: string) => void
  onOpenTranscript: (code: string) => void
  onImport: () => void
}

const MAX_NAME_LENGTH = 40

export function Home({ token, defaultName, onOpenCall, onOpenTranscript /* , onImport: used by the hidden Import card */ }: Props) {
  const [code, setCode] = useState('')
  const [callName, setCallName] = useState(() => loadCallName() || defaultName)
  const name = callName.trim()
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const start = async () => {
    setBusy(true)
    setError('')
    try {
      const { session } = await createSession(token)
      saveCallName(name)
      onOpenCall(session.code, name)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const join = (e: FormEvent) => {
    e.preventDefault()
    saveCallName(name)
    onOpenCall(code.trim().toUpperCase(), name)
  }

  const nameField = (id: string) => (
    <>
      <label htmlFor={id}>Your name in this call</label>
      <input
        id={id}
        value={callName}
        onChange={(e) => setCallName(e.target.value)}
        maxLength={MAX_NAME_LENGTH}
        autoComplete="name"
        required
      />
    </>
  )

  return (
    <div className="home-grid">
      <section className="card">
        <h2>Start a session</h2>
        <p className="muted">
          Your name is recorded on your lines. Someone on the call without VoiceBridge is recorded as Computer.
        </p>
        {nameField('start-name')}
        <button className="primary" onClick={start} disabled={busy || !name}>
          {busy ? 'Creating…' : 'Start session'}
        </button>
      </section>
      <form className="card" onSubmit={join}>
        <h2>Join a session</h2>
        <p className="muted">Enter the 6-character code the other person shared.</p>
        {nameField('join-name')}
        <label htmlFor="code">Invite code</label>
        <input
          id="code"
          className="code-input"
          value={code}
          onChange={(e) => setCode(e.target.value.toUpperCase())}
          maxLength={6}
          placeholder="K7P2QX"
          autoComplete="off"
        />
        <button className="primary" disabled={code.trim().length !== 6 || !name}>
          Join
        </button>
      </form>
      {/* Import is hidden while the app is used live; restore this card to bring it back.
      <section className="card">
        <h2>Import a recording</h2>
        <p className="muted">Stereo file, one person per channel. You pick which side is you.</p>
        <button onClick={onImport}>Import file</button>
      </section>
      */}
      <PastCalls token={token} onOpen={onOpenTranscript} />
      {error && <p className="error">{error}</p>}
    </div>
  )
}
