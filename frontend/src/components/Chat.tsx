import { useEffect, useRef, useState } from 'react'
import { keepShown, revealDelayMs } from '../lib/live'
import type { ChatMessage } from '../lib/types'

interface Props {
  messages: ChatMessage[]
  myId: string
  /** Timestamp shown as 00:00 (call start). */
  originTs: number
  /** Text shown while there are no messages (none by default). */
  emptyHint?: string
  /** In-room meeting: this speaker is on the left and everyone is shown by name (nobody is "You"). */
  leftSpeakerId?: string
}

const clock = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

const reduceMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/** A bubble's text. While live (italic) new words appear one by one; final text shows at once. */
function LiveText({ text, final }: { text: string; final: boolean }) {
  const words = text.split(/\s+/).filter(Boolean)
  const [shown, setShown] = useState(final ? words.length : 0)
  const previous = useRef<string[]>(words)

  // A new version of the line: keep the words that are still right, reveal the rest.
  useEffect(() => {
    setShown((n) => keepShown(n, previous.current, words))
    previous.current = words
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text])

  const all = final || reduceMotion()
  useEffect(() => {
    if (all || shown >= words.length) return
    const timer = setTimeout(() => setShown((n) => n + 1), revealDelayMs(words.length - shown))
    return () => clearTimeout(timer)
  }, [all, shown, words.length])

  if (all) return <span>{text}</span>
  return (
<span className="partial-text">{words.slice(0, shown).join(' ')}</span>
  )
}

/** Mirrored chat: my speech on the left, the other person on the right. */
export function Chat({ messages, myId, originTs, emptyHint = '', leftSpeakerId }: Props) {
  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  if (messages.length === 0) return <div className="chat chat-empty">{emptyHint}</div>

  return (
    <div className="chat" role="log" aria-live="polite">
      {messages.map((m) => {
        const room = leftSpeakerId !== undefined
        const left = room ? m.speakerId === leftSpeakerId : m.speakerId === myId
        return (
          <div key={m.id} className={`bubble-row ${left ? 'mine' : 'theirs'}`}>
            <div className="bubble">
              <div className="bubble-meta">
                <span>{left && !room ? 'You' : m.speakerName}</span>
                <span>{clock(m.startTs - originTs)}</span>
              </div>
              <div className="bubble-text">
                <LiveText text={m.text} final={m.final} />
              </div>
            </div>
          </div>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}
