import { useEffect, useRef } from 'react'
import { toTurns } from '../lib/turns'
import type { ChatMessage } from '../lib/types'

interface Props {
  messages: ChatMessage[]
  myId: string
  /** Timestamp shown as 00:00 (call start). */
  originTs: number
  emptyHint: string
}

const clock = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

/** Mirrored chat: my speech on the left, the other person on the right. */
export function Chat({ messages, myId, originTs, emptyHint }: Props) {
  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  if (messages.length === 0) return <div className="chat chat-empty">{emptyHint}</div>

  return (
    <div className="chat" role="log" aria-live="polite">
      {toTurns(messages).map((turn) => {
        const mine = turn.speakerId === myId
        const live = !turn.parts[turn.parts.length - 1].final
        return (
          <div key={turn.parts[0].id} className={`bubble-row ${mine ? 'mine' : 'theirs'}`}>
            <div className="bubble">
              <div className="bubble-meta">
                <span>{mine ? 'You' : turn.speakerName}</span>
                <span>{clock(turn.startTs - originTs)}</span>
              </div>
              <div className="bubble-text">
                {turn.parts.map((m, i) => (
                  <span key={m.id} className={m.final ? '' : 'partial-text'}>
                    {i > 0 && ' '}
                    {m.text}
                  </span>
                ))}
                {live && <span className="typing"> …</span>}
              </div>
            </div>
          </div>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}
