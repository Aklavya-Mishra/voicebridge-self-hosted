import { useState, type FormEvent } from 'react'
import { AGENT_DOWNLOAD } from '../lib/agent'
import { createSession } from '../lib/api'

interface Props {
  token: string
  agentFound: boolean
  onOpenCall: (code: string) => void
  onImport: () => void
}

export function Home({ token, agentFound, onOpenCall, onImport }: Props) {
  const [code, setCode] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const start = async () => {
    setBusy(true)
    setError('')
    try {
      const { session } = await createSession(token)
      onOpenCall(session.code)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const join = (e: FormEvent) => {
    e.preventDefault()
    onOpenCall(code.trim().toUpperCase())
  }

  return (
    <div className="home-grid">
      <section className="card">
        <h2>Start a session</h2>
        <p className="muted">Get an invite code and share it with the person you are calling.</p>
        <button className="primary" onClick={start} disabled={busy}>
          {busy ? 'Creating…' : 'Start session'}
        </button>
      </section>
      <form className="card" onSubmit={join}>
        <h2>Join a session</h2>
        <p className="muted">Enter the 6-character code the other person shared.</p>
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
        <button className="primary" disabled={code.trim().length !== 6}>
          Join
        </button>
      </form>
      <section className="card">
        <h2>Desktop agent</h2>
        <p className="muted">
          Transcribes any call app (Meet, Teams, Zoom…) without sharing your screen: your mic on your side, the
          computer's audio on the other side. Windows only.
        </p>
        {agentFound ? (
          <p className="agent-ok">Agent connected on this PC</p>
        ) : (
          <>
            <a className="button" href={AGENT_DOWNLOAD} download>
              Download agent (.exe)
            </a>
            <p className="muted small">Run it once; it stays in the system tray and starts with Windows.</p>
          </>
        )}
      </section>
      <section className="card">
        <h2>Import a recording</h2>
        <p className="muted">Stereo file, one person per channel. You pick which side is you.</p>
        <button onClick={onImport}>Import file</button>
      </section>
      {error && <p className="error">{error}</p>}
    </div>
  )
}
