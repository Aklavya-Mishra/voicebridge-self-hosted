import type { Domain, NbaData, NbaHint } from './types'

export const DOMAIN_OPTIONS: { value: Domain; label: string }[] = [
  { value: 'auto', label: 'Auto (detect)' },
  { value: 'sales', label: 'Sales' },
  { value: 'marketing', label: 'Marketing' },
  { value: 'telecom', label: 'Telecom' },
  { value: 'tech_support', label: 'Tech support' },
]

const label = (domain: string) => DOMAIN_OPTIONS.find((o) => o.value === domain)?.label ?? domain

/** The call type as shown in the Assist panel. */
export const domainText = (domain: Domain, detected: string | null): string =>
  domain !== 'auto' ? label(domain) : detected ? `${label(detected)} · detected` : 'Detecting call type…'

/** Newest first; a hint already shown (same id) is replaced, not repeated. */
export const addHint = (hints: NbaHint[], hint: NbaHint): NbaHint[] => [hint, ...hints.filter((h) => h.id !== hint.id)]

/** The live panel shows only the newest hints, so the advisor reads one or two, not a list. */
export const MAX_VISIBLE_HINTS = 2

export const visibleHints = (hints: NbaHint[]): NbaHint[] => hints.slice(0, MAX_VISIBLE_HINTS)

export const hintsNewestFirst = (hints: NbaHint[]): NbaHint[] => [...hints].sort((a, b) => b.ts - a.ts)

/** "current_plan" -> "Current plan"; short keys such as "os" -> "OS". */
export const factLabel = (key: string): string => {
  if (key.length <= 3) return key.toUpperCase()
  const words = key.replace(/_/g, ' ').trim()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/** Customer info for the call record: every value a fact had during the call, so nothing shown is lost. */
export const savedFacts = (nba: NbaData): Record<string, string> => {
  const out: Record<string, string> = {}
  for (const key of new Set([...Object.keys(nba.fact_log ?? {}), ...Object.keys(nba.facts)])) {
    const values = nba.fact_log?.[key]?.length ? nba.fact_log[key] : [nba.facts[key]]
    out[key] = values.filter(Boolean).join('; ')
  }
  return out
}
