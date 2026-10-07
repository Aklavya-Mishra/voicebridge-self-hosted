/** Audio helpers for the capture path: resampling and PCM16 chunking. No browser APIs, so they are unit-tested. */

export class Downsampler {
  private ratio: number
  private carry = new Float32Array(0)

  constructor(inputRate: number, targetRate: number) {
    this.ratio = inputRate / targetRate
  }

  /** Averages each window of input samples into one output sample; keeps the remainder. */
  process(input: Float32Array): Float32Array {
    const data = new Float32Array(this.carry.length + input.length)
    data.set(this.carry)
    data.set(input, this.carry.length)
    const outLength = Math.floor(data.length / this.ratio)
    const out = new Float32Array(outLength)
    for (let i = 0; i < outLength; i++) {
      const start = Math.floor(i * this.ratio)
      const end = Math.max(start + 1, Math.floor((i + 1) * this.ratio))
      let sum = 0
      for (let j = start; j < end; j++) sum += data[j]
      out[i] = sum / (end - start)
    }
    this.carry = data.slice(Math.floor(outLength * this.ratio))
    return out
  }
}

const toInt16 = (s: number) => {
  const c = Math.max(-1, Math.min(1, s))
  return c < 0 ? c * 0x8000 : c * 0x7fff
}

/**
 * Collects float samples per channel and emits PCM16 chunks of `chunkFrames` frames,
 * interleaved when there is more than one channel (stereo: left = mic, right = call audio).
 */
export class PcmChunker {
  private channels: number
  private chunkFrames: number
  private pending: number[][]

  constructor(channels: number, chunkFrames: number) {
    this.channels = channels
    this.chunkFrames = chunkFrames
    this.pending = Array.from({ length: channels }, () => [])
  }

  push(perChannel: Float32Array[]): ArrayBuffer[] {
    perChannel.forEach((samples, ch) => {
      for (const s of samples) this.pending[ch].push(s)
    })
    const out: ArrayBuffer[] = []
    while (this.pending.every((p) => p.length >= this.chunkFrames)) {
      const blocks = this.pending.map((p) => p.splice(0, this.chunkFrames))
      const pcm = new Int16Array(this.chunkFrames * this.channels)
      for (let i = 0; i < this.chunkFrames; i++) {
        for (let ch = 0; ch < this.channels; ch++) pcm[i * this.channels + ch] = toInt16(blocks[ch][i])
      }
      out.push(pcm.buffer)
    }
    return out
  }
}
