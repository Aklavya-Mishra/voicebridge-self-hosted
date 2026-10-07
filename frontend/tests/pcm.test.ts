import assert from 'node:assert/strict'
import { test } from 'node:test'
import { PcmChunker } from '../src/lib/pcm.ts'

const f32 = (...v: number[]) => Float32Array.from(v)

test('mono: emits full chunks of PCM16 and keeps the remainder', () => {
  const chunker = new PcmChunker(1, 3)
  assert.equal(chunker.push([f32(0, 0.5)]).length, 0)
  const out = chunker.push([f32(-1, 1, 0.25)])
  assert.equal(out.length, 1)
  assert.deepEqual([...new Int16Array(out[0])], [0, 16383, -32768])
})

test('stereo: interleaves left (mic) and right (call) frame by frame', () => {
  const chunker = new PcmChunker(2, 2)
  const out = chunker.push([f32(0.5, 0.5, 0.5), f32(-0.5, -0.5, -0.5)])
  assert.equal(out.length, 1)
  assert.deepEqual([...new Int16Array(out[0])], [16383, -16384, 16383, -16384])
})

test('stereo: a channel that delivers fewer samples does not shift the other', () => {
  const chunker = new PcmChunker(2, 2)
  assert.equal(chunker.push([f32(0.5, 0.5), f32(-0.5)]).length, 0)
  const out = chunker.push([f32(), f32(-0.5)])
  assert.deepEqual([...new Int16Array(out[0])], [16383, -16384, 16383, -16384])
})

test('clips samples outside -1..1', () => {
  const out = new PcmChunker(1, 2).push([f32(2, -2)])
  assert.deepEqual([...new Int16Array(out[0])], [32767, -32768])
})
