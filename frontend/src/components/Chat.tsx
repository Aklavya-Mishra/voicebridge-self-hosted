import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../lib/types'

interface Props {
  messages: ChatMessage[]
  myId: string
  /** Timestamp shown as 00:00 (call start). */
  originTs: number
  /** Text shown while there are no messages (none by default). */
  emptyHint?: string
}

const clock = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

/** Mirrored chat: my speech on the left, the other person on the right. */
export function Chat({ messages, myId, originTs, emptyHint = '' }: Props) {
  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  if (messages.length === 0) return <div className="chat chat-empty">{emptyHint}</div>

  return (
    <div className="chat" role="log" aria-live="polite">
      {messages.map((m) => {
        const mine = m.speakerId === myId
        return (
          <div key={m.id} className={`bubble-row ${mine ? 'mine' : 'theirs'}`}>
            <div className="bubble">
              <div className="bubble-meta">
                <span>{mine ? 'You' : m.speakerName}</span>
                <span>{clock(m.startTs - originTs)}</span>
              </div>
              <div className="bubble-text">
                <span className={m.final ? '' : 'partial-text'}>{m.text}</span>
                {!m.final && <span className="typing"> …</span>}
              </div>
            </div>
          </div>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}
