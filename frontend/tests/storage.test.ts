import assert from 'node:assert/strict'
import { test } from 'node:test'
import { hasConsented, loadHintsPref, saveConsent, saveHintsPref } from '../src/lib/storage.ts'

const store = new Map<string, string>()
;(globalThis as { localStorage?: unknown }).localStorage = {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => store.set(k, v),
  removeItem: (k: string) => store.delete(k),
}

test('the recording notice is agreed to once per call', () => {
  assert.equal(hasConsented('ABC234'), false)
  saveConsent('ABC234')
  assert.equal(hasConsented('ABC234'), true)
  assert.equal(hasConsented('XYZ987'), false)
})

test('the hints choice is remembered for the next call; none means the server default', () => {
  assert.equal(loadHintsPref(), null)
  saveHintsPref(false)
  assert.equal(loadHintsPref(), false)
  saveHintsPref(true)
  assert.equal(loadHintsPref(), true)
})
