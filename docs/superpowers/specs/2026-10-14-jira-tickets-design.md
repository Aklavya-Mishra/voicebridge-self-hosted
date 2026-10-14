# Jira tickets from the call (design)

Date: 2026-10-14. Status: draft for review. Scope: proof of concept.

## Goal

When anyone on the call (the advisor or the other person) says something like "create a Jira ticket where
transcription has issues", VoiceBridge drafts a Jira ticket with a suitable title and description and shows it
in the Assist panel. The advisor checks it, can edit it, and clicks **Create** (human in the loop). The ticket
is created in the advisor's Jira, and the card shows its key and a link.

Success: in a live call, saying the phrase shows a "Drafting ticket…" card within a second, a title within a
few seconds, and one click creates a real ticket (e.g. `SCRUM-5`) in `aklavyamishra.atlassian.net`.

## Decisions taken

- Either speaker can trigger it. The backend creates the ticket with the Jira account in `backend/.env`, so the
  other person needs nothing installed. Per-user Jira accounts (when both people run the app) are later work.
- Ask first: nothing is created without the advisor's click (no automatic creation, no undo needed).
- Detection: phrase match first, then the local model writes the title (approach A). No model call on lines
  that do not ask for a ticket.

## Configuration

New settings (`app/config.py`, `.env`, `.env.example`):

| Setting | Default | Meaning |
|---|---|---|
| `JIRA_URL` | empty | e.g. `https://aklavyamishra.atlassian.net` |
| `JIRA_EMAIL` | empty | Atlassian account email |
| `JIRA_API_TOKEN` | empty | API token (secret, only in `.env`) |
| `JIRA_PROJECT_KEY` | empty | e.g. `SCRUM` |
| `JIRA_ISSUE_TYPE` | `Task` | issue type name in that project |

The feature is on only when URL, email, token and project key are all set. `GET /api/health` (which already
reports `nba`) also reports `tickets: true|false`, so the web app shows the card only then.

## Backend

### `app/jira_client.py`

`JiraClient(settings, transport=None)` with one method:

`async create_issue(title, description) -> {"key": "SCRUM-5", "url": "<JIRA_URL>/browse/SCRUM-5"}`

- `POST /rest/api/3/issue` with basic auth (email + token), fields `project.key`, `issuetype.name`, `summary`
  (title, cut to 255 characters) and `description` in Atlassian Document Format (one paragraph per line).
- Timeout 15 s. A non-2xx answer raises `JiraError` with a short reason taken from Jira's `errorMessages` /
  `errors` (never the token).

### `app/tickets.py`: `TicketService`

Built in `app/main.py` when Jira is configured; it shares the hint model (`nba_llm`: same Ollama model and the
same 4096-token context, so Ollama does not reload the model).

- `on_final(code, utterance)`: called for every finished line of either speaker (from `socket_events.on_final`,
  next to the hint service; an error here never stops transcription).
  1. Match the line against `TRIGGER_PHRASES` with the whole-word fuzzy match already used by hint rules
     (`app.nba.rules._says`, threshold `nba_phrase_match`). Phrases: "create a jira ticket", "create a jira",
     "create a ticket", "raise a ticket", "raise a jira", "log a jira", "log a ticket", "open a jira",
     "open a ticket", "file a ticket", "jira ticket".
  2. No match, or a draft of this call is still open (drafting/ready): stop. One open draft per call.
  3. New draft `{id, status: "drafting", requested_by, ts}`; save and emit `ticket_draft`.
  4. Ask the model for `{title, description}` (JSON schema; title at most 80 characters, imperative, no ticket
     words such as "Create a ticket"; description 1-3 sentences) from the last 8 lines of the transcript.
     Time limit 45 s (it may wait behind a hint turn on the same model).
  5. On a model error or timeout, fall back: title from the words after the phrase ("…where transcription has
     issues" becomes "Transcription has issues"), or "Follow-up from call <code>" when nothing follows;
     description is the trigger line plus the 3 lines before it.
  6. Mask the title and description again (`mask_text`) and set `status: "ready"`; save and emit.
- `create(code, id, title, description)`: only for a draft in `ready` (or `failed`, to retry). Title must not be
  empty. Calls `JiraClient.create_issue`; on success `status: "created"` with `key` and `url`; on `JiraError`
  `status: "failed"` with `error`. Saves and emits either way. Returns the draft.
- `dismiss(code, id)`: a `ready`/`failed` draft becomes `dismissed`; saved and emitted.

### Storage

New `tickets` list on `Session` (`sessions.py`), saved in a new `tickets` TEXT column (added through
`_ADDED_COLUMNS`, so existing databases keep working) and returned in `Session.to_dict()`. Kept apart from
`session.nba`, which the hint service rewrites on every turn.

### Socket events (`socket_events.py`)

- web -> server `ticket_create {code, id, title, description}` -> ack `{ok, ticket}` or `{ok: false, error}`.
- web -> server `ticket_dismiss {code, id}` -> ack `{ok}`.
- server -> room `ticket_draft {code, id, status, title, description, requested_by, key?, url?, error?}`.
- Both web events require the sender to have joined the call. Unknown id: `{ok: false, error}`.

## Web app

- Types (`lib/types.ts`): `TicketDraft`; `Session.tickets?: TicketDraft[]`.
- `CallView` keeps `tickets` (from the joined session, updated by `ticket_draft`), passes them to `AssistPanel`
  with `onCreate` / `onDismiss` that call the socket events and show an error on a failed ack.
- `AssistPanel`: a **Jira ticket** section at the top, only when tickets are enabled and a draft exists:
  - drafting: "Drafting ticket from: '<trigger line>'…"
  - ready / failed: title input (editable), description textarea (editable), **Create ticket** and
    **Dismiss** buttons; a failed draft also shows the error.
  - created: "Created SCRUM-5" linking to the ticket (new tab), with its title.
  - Created tickets of the call stay listed (newest first); dismissed ones are hidden.
- `TranscriptView` (past call): created tickets listed under the summary with their links.

## Error handling

- Jira not configured: no trigger matching, no card, nothing else changes.
- Model slow or failing: the fallback title is used; the advisor can edit it.
- Jira error (bad token, wrong project key or issue type, network): the card shows the error and **Create**
  can be clicked again after fixing `.env` and restarting.
- Duplicate phrase while a draft is open: ignored.

## Testing

- `tests/test_tickets.py`: trigger phrases match whole words in either speaker's line ("I'll raise a ticket for
  that" yes; "tickets were sold out" no); one open draft per call; model answer becomes the draft; model
  failure and timeout use the fallback title; create success and Jira failure; dismiss; masking.
- `tests/test_jira_client.py`: request body (project, issue type, summary, ADF description, auth) and error
  reason, with `httpx.MockTransport`.
- Socket test: a spoken trigger (fake STT) produces `ticket_draft` drafting then ready; `ticket_create` with a
  fake Jira returns the key and emits created.
- Frontend: helper tests for the ticket list (order, hidden dismissed) in `tests/`.
- Manual: one real ticket created in `SCRUM` from a simulated call, with the advisor's go-ahead.

## Out of scope (later)

- Per-user Jira accounts and choosing whose Jira gets the ticket when both people run the app.
- Picking project, issue type, assignee or priority from the card.
- Understanding requests without a trigger phrase (model check on every line).
- Other tools (e.g. email, CRM) through the same draft-and-confirm card.
