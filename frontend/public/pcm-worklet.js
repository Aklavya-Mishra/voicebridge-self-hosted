// Copies each block of samples to the main thread, one Float32Array per channel.
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]
    if (input && input.length && input[0].length) this.port.postMessage(input.map((channel) => channel.slice(0)))
    return true
  }
}

registerProcessor('pcm-capture', PcmCapture)
