import type { Domain, Session, SessionSummary, User } from './types'

// Empty base means same origin (the Vite dev server proxies /api and /socket.io).
export const API_BASE: string = import.meta.env.VITE_API_URL ?? ''

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`
    throw new ApiError(res.status, detail)
  }
  return body as T
}

export const login = (name: string) =>
  request<{ token: string; user: User }>('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })

export const listSessions = (token: string) => request<{ sessions: SessionSummary[] }>('/api/sessions', {}, token)

export const fetchMe = (token: string) => request<{ user: User }>('/api/auth/me', {}, token)

export const getSession = (token: string, code: string) =>
  request<{ session: Session }>(`/api/sessions/${encodeURIComponent(code)}`, {}, token)

export const createSession = (token: string, domain: Domain = 'auto') =>
  request<{ session: Session }>(
    '/api/sessions',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ domain }) },
    token,
  )

export const importRecording = (token: string, file: File, meChannel: 'left' | 'right') => {
  const form = new FormData()
  form.append('file', file)
  form.append('me_channel', meChannel)
  return request<{ session: Session }>('/api/import', { method: 'POST', body: form }, token)
}

export const health = () => request<{ status: string; stt_provider: string; nba?: boolean }>('/api/health')
