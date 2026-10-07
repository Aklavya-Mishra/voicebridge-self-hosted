import { useState, type FormEvent } from 'react'

interface Props {
  name: string
  /** Shown before the name, e.g. "You · ". */
  prefix?: string
  className?: string
  /** Saves the new name; throw to show an error and stay in edit mode. */
  onSave: (name: string) => Promise<void>
}

const MAX_NAME_LENGTH = 40

/** A speaker's name with an edit (✎) button that turns it into an inline Save / Cancel form. */
export function EditableName({ name, prefix = '', className = '', onSave }: Props) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(name)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (!editing) {
    return (
      <span className={className}>
        {prefix}
        {name}
        <button
          className="icon-button"
          title="Edit name"
          aria-label={`Edit name ${name}`}
          onClick={() => {
            setValue(name)
            setError('')
            setEditing(true)
          }}
        >
          ✎
        </button>
      </span>
    )
  }

  const save = async (e: FormEvent) => {
    e.preventDefault()
    const next = value.trim()
    if (!next || next === name) {
      setEditing(false)
      return
    }
    setBusy(true)
    setError('')
    try {
      await onSave(next)
      setEditing(false)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className={`${className} name-edit`} onSubmit={save}>
      {prefix}
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Escape' && setEditing(false)}
        maxLength={MAX_NAME_LENGTH}
        style={{ width: `${value.length + 3}ch` }}
        aria-label="Name"
        autoFocus
      />
      <button
        type="submit"
        className="icon-button save"
        title="Save"
        aria-label="Save name"
        disabled={busy || !value.trim()}
      >
        ✓
      </button>
      <button type="button" className="icon-button" title="Cancel" aria-label="Cancel" onClick={() => setEditing(false)}>
        ✕
      </button>
      {error && <span className="error small">{error}</span>}
    </form>
  )
}
