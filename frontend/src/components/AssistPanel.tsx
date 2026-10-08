import { DOMAIN_OPTIONS, domainText, factLabel, visibleHints } from '../lib/assist'
import type { Domain, NbaHint, NbaStatusEvent } from '../lib/types'

interface Props {
  domain: Domain
  detected: string | null
  hints: NbaHint[]
  facts: Record<string, string>
  status: NbaStatusEvent['state']
  reason?: string
  onDomainChange: (domain: Domain) => void
}

/** Live hints for the advisor (newest first) and what the customer has told so far. */
export function AssistPanel({ domain, detected, hints, facts, status, reason, onDomainChange }: Props) {
  return (
    <aside className="assist card" aria-label="Call assist">
      <div className="assist-head">
        <h2>Assist</h2>
        <select aria-label="Call type" value={domain} onChange={(e) => onDomainChange(e.target.value as Domain)}>
          {DOMAIN_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      <p className="muted small">{domainText(domain, detected)}</p>
      {status === 'thinking' && <p className="muted small">Thinking…</p>}
      {status === 'unavailable' && (
        <p className="notice small">AI hints unavailable ({reason}). Rule hints still work.</p>
      )}
      <h3>Next best action</h3>
      <HintList hints={visibleHints(hints)} />
      <h3>Customer info</h3>
      <FactList facts={facts} />
    </aside>
  )
}

export function HintList({ hints }: { hints: NbaHint[] }) {
  if (hints.length === 0) return <p className="muted small">Hints appear after the other person speaks.</p>
  return (
    <ol className="hints">
      {hints.map((h, i) => (
        <li key={h.id} className={`hint ${h.priority}${i === 0 ? ' latest' : ''}`}>
          {h.kind === 'rule' && <span className="hint-tag">rule</span>}
          {h.text}
        </li>
      ))}
    </ol>
  )
}

export function FactList({ facts }: { facts: Record<string, string> }) {
  const entries = Object.entries(facts)
  if (entries.length === 0) return <p className="muted small">Nothing yet.</p>
  return (
    <dl className="facts">
      {entries.map(([key, value]) => (
        <div key={key} className="fact">
          <dt>{factLabel(key)}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}
