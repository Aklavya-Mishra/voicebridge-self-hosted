import { useCallback, useEffect, useState } from 'react'
import { CallView } from './components/CallView'
import { Home } from './components/Home'
import { ImportView } from './components/ImportView'
import { Login } from './components/Login'
import { connectAgent } from './lib/agent'
import { health } from './lib/api'
import { loadAuth, saveAuth, type StoredAuth } from './lib/storage'

type View = { name: 'home' } | { name: 'call'; code: string } | { name: 'import' }

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
          <CallView code={view.code} token={auth.token} me={auth.user} onLeave={goHome} onAuthExpired={signOut} />
        ) : view.name === 'import' ? (
          <ImportView token={auth.token} me={auth.user} onBack={goHome} />
        ) : (
          <Home
            token={auth.token}
            agentFound={agentFound}
            onOpenCall={(code) => setView({ name: 'call', code })}
            onImport={() => setView({ name: 'import' })}
          />
        )}
      </main>
    </div>
  )
}
