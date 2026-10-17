# Slack recap at call end (design)

Date: 2026-10-17. Status: draft for review. Scope: proof of concept, first step of "the call does the follow-up"
(Jira ticket done; Slack recap now; calendar, sprint dates, decision notes later, one by one).

## Goal

When a call (or an in-room meeting) ends and VoiceBridge has written its summary, the recap is posted to Slack
straight away, as the person who recorded the call: their name and photo, not a bot. People in the channel see
which call it was, who was on it, what was decided, what is next, and links to anything done during the call
(Jira tickets).

Success: end a live call; within a few seconds of the summary showing in VoiceBridge, a message from
**Aklavya** appears in `#voicebridge-recaps` of the `voicebridgegroup.slack.com` workspace, and the call screen shows
"Recap sent to #voicebridge-recaps" with a link to it.

## Decisions taken (by the user)

- **Posts as the user, not as a bot.** For the demo only one person uses the app, so one Slack user token in
  `backend/.env` is enough. Each person connecting their own Slack ("Connect Slack" sign-in, one token per user)
  is later work, before shipping to the org.
- **Sent right away**, no review or Send click: "that'll be transparent".
- **Content stays light**: identify the call and name the people; the rest comes from the existing summary.
- Slack first (free plan, personal workspace). Teams is a later target with the same input (Workflows webhook or
  Graph); the free Teams plan has no Workflows, so it can't be tested yet.
- No LangGraph or other agent framework: this is one function at call end.

## Slack setup (done 2026-10-17, plus one step left)

- Slack app **VoiceBridge** (`A0CA4912V8U`) exists in workspace `VoiceBridge` (`T0C9K5HQVE3`) with bot scopes
  `chat:write`, `chat:write.public`, `im:write`, `users:read`, `channels:read`. Bot token is in `.env` as
  `SLACK_BOT_TOKEN` (checked with `auth.test`).
- **Left to do:** add user scope `chat:write` to the app (OAuth & Permissions, User Token Scopes), reinstall,
  and paste the **User OAuth Token** (`xoxp-...`) into `.env` as `SLACK_USER_TOKEN`. The user does the paste
  (tokens are never read or typed by the assistant).

## Configuration

New settings (`app/config.py`, `.env`, `.env.example`):

| Setting | Default | Meaning |
|---|---|---|
| `SLACK_USER_TOKEN` | empty | `xoxp-...`; recaps are posted with it, so they show as that person |
| `SLACK_BOT_TOKEN` | empty | `xoxb-...`; used only to look people up for @mentions |
| `SLACK_RECAP_CHANNEL` | empty | channel name, e.g. `voicebridge-recaps` (with or without `#`) |

The recap is on only when `SLACK_USER_TOKEN` and `SLACK_RECAP_CHANNEL` are set. Without `SLACK_BOT_TOKEN`
names are plain text (no @mentions). `GET /api/health` also reports `slack_recap: true|false`.

## What is posted

Built from the session's summary when its status becomes `ready`:

- **Header**: for a call, `Call recap: <purpose>`; for an in-room meeting, `Meeting minutes: <first sentence
  of summary>`. Then the date and time the call started (local time of the server) and its length.
- **People**: everyone who spoke, by the name on their lines (renamed names included). A name becomes a Slack
  @mention when exactly one workspace member matches it (real name or display name, case-insensitive, whole
  first name or full name); otherwise it stays plain text.
- **Summary**: the summary text.
- **Call**: `Outcome` and `Action items`. **Meeting**: `Decisions`, `Action items`, `Open issues`.
  Empty sections are left out.
- **Done during the call**: each created Jira ticket as `<key>: <title>` linking to the ticket.
- Footer: "Posted by VoiceBridge".

Uses Slack Block Kit (header, section, context blocks) with a plain `text` fallback for notifications.
PII masking already applied to the summary carries over; nothing from the raw transcript is posted.

## When it is sent, and updates

- After each end-of-call analysis that finishes with status `ready`, for live sessions (`kind` live, both
  `call` and `room` modes). Imported recordings are not posted.
- The first time: `chat.postMessage`. The channel id and message `ts` are kept on the session (new field
  `recap`, saved with the session in SQLite like `tickets`).
- If the summary is written again later (a ticket created after the summary, recording started and stopped
  again, or a speaker renamed), the same message is edited with `chat.update`, never a second post.
- Status `unavailable` (no model reachable): nothing is posted.

`session.recap`: `{"status": "sent" | "failed", "channel": "C…", "ts": "…", "permalink": "…", "error": "…"}`.

## Failures

- Slack errors (bad token, channel not found, network) never affect the call, the transcript or the summary.
  The recap status becomes `failed` with Slack's error code (e.g. `channel_not_found`, `invalid_auth`), logged
  once.
- The call screen's summary area shows one line:
  - sent: "Recap sent to #voicebridge-recaps" with an **Open in Slack** link (permalink);
  - failed: "Recap not sent to Slack: <reason>" with a **Retry** button (re-sends: update if a `ts` exists,
    else post).
  The same line shows on the past-call screen.
- A timeout of 10 s per Slack call.

## Backend

- `app/slack_client.py`: thin async client over Slack Web API (`chat.postMessage`, `chat.update`,
  `chat.getPermalink`, `users.list`, `conversations.list` to turn a channel name into an id). Uses the existing
  `aiohttp`/`httpx` dependency; no new package.
- `app/recap.py`: builds the Block Kit message from a session (pure function, unit-tested) and sends or edits
  it; resolves names to members (member list cached for 10 minutes).
- Hook: where `analyse_session` finishes with a `ready` summary, call the recap sender; it emits the updated
  session (`recap` field) to the call's room so the open call screen updates.
- Socket event `retry_recap {code}` -> ack `{ok, error?}`.

## Frontend

- Summary area of the call screen and past-call screen: the status line above, shown only when
  `health.slack_recap` is true and the session has a `recap` (or a ready summary without one yet: "Sending
  recap to Slack…").
- Matches the existing indigo palette and WCAG AA.

## Tests

- `recap.py` message building: call vs meeting sections, empty sections dropped, ticket links, @mention vs
  plain name (0, 1, 2 matches), masked values stay masked.
- Sender: first post stores `ts`; second analysis edits; Slack error -> `failed` with reason; retry; off when
  not configured; imports skipped. Slack calls mocked.
- Live check: a real call with Deepgram and the local model, recap seen in `#voicebridge-recaps` as Aklavya,
  then a ticket created after the summary edits the same message.

## Non-goals (now)

- Per-person Slack sign-in ("Connect Slack"); posting as anyone but the one configured user.
- Choosing the channel per call; DMs to attendees; threads.
- "Send a message to Balram" during the call (next action, same Slack app).
- Teams recap.

## Calls made by the assistant (check while reading)

- Header uses the summary's purpose (call) or its first sentence (meeting); there is no call title today.
- Speaker rename after posting edits the message too.
- Imports are not posted.
- @mention only on exactly one match, else plain name.
- Channel set in `.env`, one for all calls.
