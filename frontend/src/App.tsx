import { useCallback, useEffect, useState } from 'react'
import { CallView } from './components/CallView'
import { Home } from './components/Home'
import { ImportView } from './components/ImportView'
import { TranscriptView } from './components/TranscriptView'
import { Login } from './components/Login'
import { connectAgent } from './lib/agent'
import { ApiError, fetchMe, health } from './lib/api'
import { loadAuth, saveAuth, type StoredAuth } from './lib/storage'

type View =
  | { name: 'home' }
  | { name: 'call'; code: string; callName: string }
  | { name: 'transcript'; code: string }
  | { name: 'import' }

export default function App() {
  const [auth, setAuth] = useState<StoredAuth | null>(() => loadAuth())
  const [view, setView] = useState<View>({ name: 'home' })
  const [provider, setProvider] = useState('')
  const [agentFound, setAgentFound] = useState(false)

  useEffect(() => {
    health()
      .then((h) => setProvider(h.stt_provider))
      .catch(() => setProvider('offline'))
  }, [])

  // Keep handing the sign-in to a desktop agent on this PC; it may be started after the page.
  const token = auth?.token
  useEffect(() => {
    if (!token) return
    let cancelled = false
    const tryAgent = async () => {
      const found = await connectAgent(token)
      if (!cancelled) setAgentFound(found)
    }
    void tryAgent()
    const timer = setInterval(tryAgent, 10000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [token])

  const signIn = (next: StoredAuth) => {
    saveAuth(next)
    setAuth(next)
  }
  const signOut = useCallback(() => {
    saveAuth(null)
    setAuth(null)
    setView({ name: 'home' })
  }, [])
  const goHome = useCallback(() => setView({ name: 'home' }), [])

  // The server decides who a token belongs to; correct the saved identity if it differs (e.g. a sign-in
  // saved before user ids were derived from the name), or "mine" vs "theirs" in the chat would be wrong.
  useEffect(() => {
    if (!token) return
    fetchMe(token)
      .then(({ user }) =>
        setAuth((prev) => {
          if (!prev || prev.token !== token || (prev.user.id === user.id && prev.user.name === user.name)) return prev
          const next = { token, user }
          saveAuth(next)
          return next
        }),
      )
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) signOut()
      })
  }, [token, signOut])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/favicon.svg" alt="" width={24} height={24} /> VoiceBridge
        </div>
        <div className="topbar-right">
          {provider && <span className={`pill ${provider === 'offline' ? 'bad' : ''}`}>STT: {provider}</span>}
          {auth && (
            <>
              <span className="muted">{auth.user.name}</span>
              <button className="link" onClick={signOut}>
                Sign out
              </button>
            </>
          )}
        </div>
      </header>
      <main>
        {!auth ? (
          <Login onLogin={signIn} />
        ) : view.name === 'call' ? (
          <CallView
            code={view.code}
            callName={view.callName}
            token={auth.token}
            me={auth.user}
            onLeave={goHome}
            onAuthExpired={signOut}
          />
        ) : view.name === 'transcript' ? (
          <TranscriptView code={view.code} token={auth.token} onBack={goHome} />
        ) : view.name === 'import' ? (
          <ImportView token={auth.token} me={auth.user} onBack={goHome} />
        ) : (
          <Home
            token={auth.token}
            agentFound={agentFound}
            defaultName={auth.user.name}
            onOpenCall={(code, callName) => setView({ name: 'call', code, callName })}
            onOpenTranscript={(code) => setView({ name: 'transcript', code })}
            onImport={() => setView({ name: 'import' })}
          />
        )}
      </main>
    </div>
  )
}
