import { useState, type FormEvent } from 'react'
import { login } from '../lib/api'
import type { StoredAuth } from '../lib/storage'

// For now the app is used by one person; pre-fill their name.
const DEFAULT_NAME = 'Aklavya'

export function Login({ onLogin }: { onLogin: (auth: StoredAuth) => void }) {
  const [name, setName] = useState(DEFAULT_NAME)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      onLogin(await login(name))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="card narrow" onSubmit={submit}>
      <h1>Sign in</h1>
      <p className="muted">Your name is shown to the other person on the call.</p>
      <label htmlFor="name">Display name</label>
      <input id="name" value={name} onChange={(e) => setName(e.target.value)} maxLength={40} autoFocus required />
      {error && <p className="error">{error}</p>}
      <button className="primary" disabled={busy || !name.trim()}>
        {busy ? 'Signing in…' : 'Continue'}
      </button>
    </form>
  )
}
