import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { io, type Socket } from 'socket.io-client'
import { API_BASE } from '../lib/api'
import { addHint, hintsNewestFirst } from '../lib/assist'
import { MicCapture } from '../lib/mic'
import { formatTranscript } from '../lib/transcript'
import {
  fromUtterance,
  type ChatMessage,
  type Domain,
  type NbaDomainEvent,
  type NbaHint,
  type NbaStatusEvent,
  type Session,
  type SttStatusEvent,
  type TranscriptEvent,
  type User,
} from '../lib/types'
import { AssistPanel } from './AssistPanel'
import { Chat } from './Chat'
import { EditableName } from './EditableName'

interface Props {
  code: string
  token: string
  me: User
  /** Name entered for this call; recorded on this user's lines. */
  callName: string
  /** Live hints are on (backend NBA_ENABLED). */
  nbaEnabled: boolean
  onLeave: () => void
  onAuthExpired: () => void
}

type Status = 'connecting' | 'joined' | 'error'

export function CallView({ code, callName, nbaEnabled, token, me, onLeave, onAuthExpired }: Props) {
  const [status, setStatus] = useState<Status>('connecting')
  const [error, setError] = useState('')
  const [participants, setParticipants] = useState<User[]>([])
  const [messages, setMessages] = useState<Map<string, ChatMessage>>(new Map())
  const [originTs, setOriginTs] = useState(() => Date.now() / 1000)
  const [recording, setRecording] = useState(false)
  const [consentOpen, setConsentOpen] = useState(false)
  const [leaveOpen, setLeaveOpen] = useState(false)
  const [levels, setLevels] = useState<number[]>([])
  const [agentOnline, setAgentOnline] = useState(false)
  const [viaAgent, setViaAgent] = useState(false)
  const [copied, setCopied] = useState(false)
  // Speakers whose transcription fell back from Deepgram to the local engine: id -> name.
  const [localEngine, setLocalEngine] = useState<Map<string, string>>(new Map())
  // Renamed speakers who are not participants (the call audio): id -> name.
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({})
  const [domain, setDomain] = useState<Domain>('auto')
  const [detected, setDetected] = useState<string | null>(null)
  const [hints, setHints] = useState<NbaHint[]>([])
  const [facts, setFacts] = useState<Record<string, string>>({})
  const [nbaStatus, setNbaStatus] = useState<{ state: NbaStatusEvent['state']; reason?: string }>({ state: 'idle' })
  const socketRef = useRef<Socket | null>(null)
  const micRef = useRef<MicCapture | null>(null)

  const upsert = useCallback((event: TranscriptEvent) => {
    setMessages((prev) => {
      const next = new Map(prev)
      if (event.final && !event.text) next.delete(event.utterance_id)
      else
        next.set(event.utterance_id, {
          id: event.utterance_id,
          speakerId: event.speaker.id,
          speakerName: event.speaker.name,
          text: event.text,
          startTs: event.start_ts,
          endTs: event.end_ts,
          final: event.final,
        })
      return next
    })
  }, [])

  useEffect(() => {
    const socket = io(API_BASE || undefined, { auth: { token }, transports: ['websocket'] })
    socketRef.current = socket

    socket.on('connect', async () => {
      const ack = await socket.emitWithAck('join_session', { code, name: callName })
      if (!ack.ok) {
        setStatus('error')
        setError(ack.error)
        return
      }
      const session: Session = ack.session
      setParticipants(session.participants)
      setOriginTs(session.created_at)
      setMessages(new Map(session.transcript.map((u) => [u.id, fromUtterance(u)])))
      setSpeakerNames(session.speaker_names ?? {})
      setDomain(session.domain ?? 'auto')
      setDetected(session.nba?.detected ?? null)
      setHints(hintsNewestFirst(session.nba?.hints ?? []))
      setFacts(session.nba?.facts ?? {})
      setStatus('joined')
      setError('') // e.g. "Cannot reach the server" from before the connection recovered
    })
    socket.on('connect_error', (err) => {
      if (err.message === 'unauthorized') onAuthExpired()
      else {
        setStatus('error')
        setError('Cannot reach the server. Retrying…')
      }
    })
    socket.on('participants', (data: { participants: User[] }) => setParticipants(data.participants))
    socket.on('transcript', upsert)
    socket.on('speaker_renamed', (data: { speaker_id: string; name: string }) => {
      setSpeakerNames((prev) => ({ ...prev, [data.speaker_id]: data.name }))
      setMessages(
        (prev) =>
          new Map(
            [...prev].map(([id, m]) => [id, m.speakerId === data.speaker_id ? { ...m, speakerName: data.name } : m]),
          ),
      )
    })
    socket.on('agent_status', (data: { online: boolean }) => setAgentOnline(data.online))
    socket.on('stt_status', (data: SttStatusEvent) =>
      setLocalEngine((prev) => {
        const next = new Map(prev)
        if (data.engine === 'local') next.set(data.speaker.id, data.speaker.name)
        else next.delete(data.speaker.id)
        return next
      }),
    )
    socket.on('nba_hint', (hint: NbaHint) => setHints((prev) => addHint(prev, hint)))
    socket.on('nba_facts', (data: { facts: Record<string, string> }) => setFacts(data.facts))
    socket.on('nba_domain', (data: NbaDomainEvent) => {
      setDomain(data.auto ? 'auto' : ((data.domain ?? 'auto') as Domain))
      setDetected(data.auto ? data.domain : null)
    })
    socket.on('nba_status', (data: NbaStatusEvent) => setNbaStatus({ state: data.state, reason: data.reason }))

    return () => {
      void micRef.current?.stop()
      micRef.current = null
      socket.disconnect()
      socketRef.current = null
    }
  }, [code, callName, token, upsert, onAuthExpired])

  const startMic = async () => {
    setConsentOpen(false)
    setError('')
    const mic = new MicCapture()
    try {
      await mic.start((pcm) => socketRef.current?.emit('audio_chunk', { code, audio: pcm }), setLevels)
      micRef.current = mic
      setRecording(true)
      // A new mic session tries Deepgram again.
      setLocalEngine((prev) => {
        const next = new Map(prev)
        next.delete(me.id)
        return next
      })
    } catch (err) {
      await mic.stop()
      setError(`Microphone unavailable: ${(err as Error).message}`)
    }
  }

  const startAgent = async () => {
    setConsentOpen(false)
    setError('')
    const ack = await socketRef.current?.emitWithAck('agent_start', { code })
    if (!ack?.ok) {
      setError(ack?.error ?? 'Desktop agent could not start')
      return
    }
    setViaAgent(true)
    setRecording(true)
  }

  const stopMic = async () => {
    if (viaAgent) {
      await socketRef.current?.emitWithAck('agent_stop', { code })
      setViaAgent(false)
    } else {
      await micRef.current?.stop()
      micRef.current = null
      socketRef.current?.emit('stop_audio', { code })
    }
    setRecording(false)
    setLevels([])
  }

  const changeDomain = async (next: Domain) => {
    const ack = await socketRef.current?.emitWithAck('set_domain', { code, domain: next })
    if (!ack?.ok) setError(ack?.error ?? 'Could not change the call type')
  }

  const leave = async () => {
    await stopMic()
    socketRef.current?.emit('leave_session', { code })
    onLeave()
  }

  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard blocked; the code is visible on screen
    }
  }

  const sorted = useMemo(() => [...messages.values()].sort((a, b) => a.startTs - b.startTs), [messages])
  const other = participants.find((p) => p.id !== me.id)
  // The call-audio side (desktop agent): "Computer" until renamed.
  const callSpeakerId = `${me.id}-call`
  const callSideName = speakerNames[callSpeakerId] ?? 'Computer'
  const hasCallSide = viaAgent || sorted.some((m) => m.speakerId === callSpeakerId)

  /** Renames a speaker for the whole call: past lines, new lines, the saved transcript and summary. */
  const rename = async (speakerId: string, name: string) => {
    const ack = await socketRef.current?.emitWithAck('rename_speaker', { code, speaker_id: speakerId, name })
    if (!ack?.ok) throw new Error(ack?.error ?? 'Could not rename')
  }
  const myName = participants.find((p) => p.id === me.id)?.name ?? callName

  const downloadTranscript = () => {
    const text = formatTranscript(sorted)
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `transcript-${code}-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="call">
      <header className="call-header card">
        <div>
          <div className="muted small">Invite code</div>
          <button className="code-badge" onClick={copyCode} aria-label={`Invite code ${code}, copy`}>
            {code} <span className="small">{copied ? 'copied' : 'copy'}</span>
          </button>
        </div>
        <div className="people">
          <EditableName className="person mine" prefix="You · " name={myName} onSave={(n) => rename(me.id, n)} />
          {other ? (
            <span className="person theirs">{other.name}</span>
          ) : hasCallSide ? (
            <EditableName className="person theirs" name={callSideName} onSave={(n) => rename(callSpeakerId, n)} />
          ) : (
            <span className="person theirs waiting">Waiting for the other person…</span>
          )}
        </div>
        <div className="call-actions">
          {recording ? (
            <button className="danger" onClick={stopMic}>
              <span className="rec-dot" /> Stop mic
            </button>
          ) : (
            <button className="primary" onClick={() => setConsentOpen(true)} disabled={status !== 'joined'}>
              Start mic
            </button>
          )}
          <button onClick={downloadTranscript} disabled={sorted.length === 0}>
            Transcript
          </button>
          <button onClick={() => setLeaveOpen(true)}>Leave</button>
        </div>
      </header>

      {recording &&
        levels.map((peak, ch) => (
          <div key={ch} className="level-row">
            <span className="muted small">Your mic</span>
            <div className="level" aria-label="Microphone level">
              <div className="level-fill" style={{ width: `${Math.min(100, peak * 140)}%` }} />
            </div>
          </div>
        ))}
      {error && <p className="error">{error}</p>}
      {[...localEngine].map(([id, name]) => (
        <p key={id} className="notice">
          Deepgram unavailable for {id === me.id ? 'you' : name} — using local transcription.
        </p>
      ))}
      {status === 'connecting' && <p className="muted">Connecting…</p>}

      <div className={nbaEnabled ? 'call-body' : 'call-body solo'}>
        {nbaEnabled && (
          <AssistPanel
            domain={domain}
            detected={detected}
            hints={hints}
            facts={facts}
            status={nbaStatus.state}
            reason={nbaStatus.reason}
            onDomainChange={changeDomain}
          />
        )}
        <Chat
          messages={sorted}
          myId={me.id}
          originTs={originTs}
          emptyHint={recording ? 'Listening… start talking.' : 'Start your mic to transcribe your side of the call.'}
        />
      </div>

      {leaveOpen && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="leave-title">
          <div className="card modal">
            <h2 id="leave-title">Leave this call?</h2>
            <p>{recording ? 'Transcription will stop.' : 'You can download the transcript before you go.'}</p>
            <div className="modal-actions">
              <button onClick={() => setLeaveOpen(false)} autoFocus>
                No
              </button>
              <button className="danger" onClick={leave}>
                Yes, leave
              </button>
            </div>
          </div>
        </div>
      )}

      {consentOpen && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="card modal">
            <h2>Before you start</h2>
            <p>
              Your microphone will be transcribed and shared with the other person in this session. Recording or
              transcribing calls can require consent from everyone on the call.
            </p>
            {agentOnline ? (
              <p className="muted small">
                The VoiceBridge Agent captures your mic (your side) and your computer's audio (the other person),
                from any call app. Nothing is shared. Use a headset so their voice is not picked up by your mic too.
              </p>
            ) : (
              <p className="notice">
                The VoiceBridge Agent is not running on this PC, so only your mic can be transcribed. Start{' '}
                <strong>VoiceBridge Agent</strong> (or install it from the home page) to also transcribe the other
                person.
              </p>
            )}
            <div className="modal-actions">
              <button onClick={() => setConsentOpen(false)}>Cancel</button>
              {agentOnline ? (
                <button className="primary" onClick={startAgent}>
                  Start
                </button>
              ) : (
                <button onClick={startMic}>Mic only</button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
