import type { User } from './types'

const KEY = 'voicebridge.auth'
const STALE_TEST_NAMES = ['Asha']

export interface StoredAuth {
  token: string
  user: User
}

// Storage can be unavailable (private windows, blocked site data); the app works without it.
export function loadAuth(): StoredAuth | null {
  try {
    const raw = localStorage.getItem(KEY)
    const auth = raw ? (JSON.parse(raw) as StoredAuth) : null
    // One-time cleanup: "Asha" was a test sign-in; drop it so the user signs in as themselves.
    if (auth && STALE_TEST_NAMES.includes(auth.user.name)) {
      localStorage.removeItem(KEY)
      return null
    }
    return auth
  } catch {
    return null
  }
}

export function saveAuth(auth: StoredAuth | null): void {
  try {
    if (auth) localStorage.setItem(KEY, JSON.stringify(auth))
    else localStorage.removeItem(KEY)
  } catch {
    // ignore
  }
}
