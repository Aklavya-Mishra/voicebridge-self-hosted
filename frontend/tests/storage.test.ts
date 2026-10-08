import assert from 'node:assert/strict'
import { test } from 'node:test'
import { hasConsented, saveConsent } from '../src/lib/storage.ts'

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
