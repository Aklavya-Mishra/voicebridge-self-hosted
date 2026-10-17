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

const CALL_NAME_KEY = 'voicebridge.callName'

/** The name last entered for a call, to pre-fill the next one. */
export function loadCallName(): string {
  try {
    return localStorage.getItem(CALL_NAME_KEY) ?? ''
  } catch {
    return ''
  }
}

export function saveCallName(name: string): void {
  try {
    localStorage.setItem(CALL_NAME_KEY, name)
  } catch {
    // ignore
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

const HINTS_KEY = 'voicebridge.hints'

/** Whether this browser last wanted live hints on (true), off (false), or never chose (null: the server default). */
export function loadHintsPref(): boolean | null {
  try {
    const raw = localStorage.getItem(HINTS_KEY)
    return raw === null ? null : raw === 'on'
  } catch {
    return null
  }
}

export function saveHintsPref(on: boolean): void {
  try {
    localStorage.setItem(HINTS_KEY, on ? 'on' : 'off')
  } catch {
    // ignore
  }
}

const CONSENT_KEY = 'voicebridge.consented'

/** Whether the recording notice was already agreed to for this call: it is shown once per call. */
export function hasConsented(code: string): boolean {
  try {
    return (JSON.parse(localStorage.getItem(CONSENT_KEY) ?? '[]') as string[]).includes(code)
  } catch {
    return false
  }
}

export function saveConsent(code: string): void {
  try {
    const codes = (JSON.parse(localStorage.getItem(CONSENT_KEY) ?? '[]') as string[]).filter((c) => c !== code)
    localStorage.setItem(CONSENT_KEY, JSON.stringify([...codes, code].slice(-50)))
  } catch {
    // ignore
  }
}
