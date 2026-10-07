/**
 * Browser microphone capture: mono, 16 kHz PCM16 in 100 ms chunks. Used when the desktop agent is not
 * running; the agent captures the mic and the computer's audio (the other person) itself.
 */
import { Downsampler, PcmChunker } from './pcm'

export const TARGET_RATE = 16000
const CHUNK_FRAMES = 1600 // 100 ms

export class MicCapture {
  private context: AudioContext | null = null
  private stream: MediaStream | null = null
  private node: AudioWorkletNode | null = null

  /** onChunk gets PCM16 mono; onLevel gets the peak of each block (one channel). */
  async start(onChunk: (pcm: ArrayBuffer) => void, onLevel: (peaks: number[]) => void): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
    this.context = new AudioContext()
    await this.context.audioWorklet.addModule('/pcm-worklet.js')
    // Explicit mono: a stereo mic is mixed down.
    this.node = new AudioWorkletNode(this.context, 'pcm-capture', {
      channelCount: 1,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers',
    })
    this.context.createMediaStreamSource(this.stream).connect(this.node)

    const downsampler = new Downsampler(this.context.sampleRate, TARGET_RATE)
    const chunker = new PcmChunker(1, CHUNK_FRAMES)
    this.node.port.onmessage = (event: MessageEvent<Float32Array[]>) => {
      const samples = downsampler.process(event.data[0])
      onLevel([samples.reduce((peak, v) => Math.max(peak, Math.abs(v)), 0)])
      for (const pcm of chunker.push([samples])) onChunk(pcm)
    }
    // The worklet must be connected to run; route it through a muted gain so nothing is played back.
    const mute = this.context.createGain()
    mute.gain.value = 0
    this.node.connect(mute).connect(this.context.destination)
  }

  async stop(): Promise<void> {
    this.node?.port.close()
    this.node?.disconnect()
    this.stream?.getTracks().forEach((t) => t.stop())
    await this.context?.close().catch(() => undefined)
    this.node = null
    this.stream = null
    this.context = null
  }
}
