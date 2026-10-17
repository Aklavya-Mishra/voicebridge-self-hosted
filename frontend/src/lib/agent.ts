import { API_BASE } from './api'

/** The desktop agent listens here on the same PC (see agent/local_api.py). */
const AGENT_URL = 'http://127.0.0.1:47801'
export const AGENT_DOWNLOAD = '/downloads/VoiceBridge-SelfHosted-Agent.exe'

/**
 * Hands the agent this sign-in so it can connect to VoiceBridge as the same user.
 * Returns false when no agent is running on this PC.
 */
export async function connectAgent(token: string): Promise<boolean> {
  try {
    const res = await fetch(`${AGENT_URL}/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server: API_BASE || window.location.origin, token }),
    })
    return res.ok
  } catch {
    return false
  }
}
