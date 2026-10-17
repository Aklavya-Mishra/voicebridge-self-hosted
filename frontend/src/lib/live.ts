/** Live text: words are revealed one at a time, so Deepgram's updates (about one a second) read as speech. */

/** How many leading words two versions of a live line share. */
export const sharedWords = (a: string[], b: string[]): number => {
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  return i
}

/** Words to show after a new version of the text arrives: keep what is still right, never more than the text. */
export const keepShown = (shown: number, before: string[], after: string[]): number =>
  Math.min(shown, sharedWords(before, after))

/** Delay between two revealed words: catch up within about a second, never slower than speech. */
export const revealDelayMs = (waiting: number): number => Math.max(40, Math.min(160, Math.floor(900 / Math.max(1, waiting))))
