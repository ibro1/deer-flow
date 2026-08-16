# Remote Control — terminal coding-agent sessions in the DeerFlow UI

The same idea as Claude Code's `/remote-control`, self-hosted and
agent-agnostic: start a coding agent in any terminal (Claude Code, opencode,
openclaude, aider, …), bridge it with the `deer-remote` CLI, and watch /
continue the session from **Workspace → Remote control** in the DeerFlow web
UI — including from your phone.

```
┌─────────────┐  stream-json / pty   ┌───────────────────┐    WebSocket    ┌──────────────────────┐
│ claude code │ ──► deer-remote ───► │ DeerFlow gateway  │ ◄─────────────► │ /workspace/          │
│ opencode …  │ ◄── (bridge CLI) ◄── │ /api/remote-ctrl  │                 │   remote-control     │
└─────────────┘   input injection    └───────────────────┘                 └──────────────────────┘
```

## Server setup

Set one environment variable on the backend:

```bash
REMOTE_CONTROL_TOKEN=$(openssl rand -hex 24)   # shared secret for bridges
```

If unset, the agent endpoint is disabled (browsers can still see historical
sessions). Transcripts persist in a dedicated SQLite file —
`REMOTE_CONTROL_DB` (default `.deer-flow/remote_control.db`, the same
project state dir the docker deployments mount as a volume, so history
survives container restarts/redeploys).

Browser access uses normal DeerFlow cookie auth; the WebSocket endpoints
replicate cookie resolution and Origin checks the same way the browser-stream
feature does.

## Client setup (any machine you code from)

```bash
pip install websocket-client
cp scripts/deer-remote ~/.local/bin/ && chmod +x ~/.local/bin/deer-remote

# in ~/.bashrc
export DEER_REMOTE_URL=https://your-deerflow-host   # DeerFlow origin
export DEER_REMOTE_TOKEN=<REMOTE_CONTROL_TOKEN>
```

## Usage

**Claude Code — structured mode** (full messages, tool calls, results):

```bash
cd ~/my-project
deer-remote claude
deer-remote claude -- --model sonnet
```

Uses `claude -p --input-format stream-json --output-format stream-json
--verbose`, the officially supported bidirectional streaming interface.
Type in the terminal *or* from the web page — both are injected as user
turns and both sides see everything.

Headless `-p` mode has no TTY to answer permission prompts, so
`deer-remote claude` defaults to `--permission-mode acceptEdits` (file
writes/edits auto-approved; other actions still gate normally). Override with
your own `--permission-mode <mode>`, or for full unattended trust on a
single-tenant box:

```bash
deer-remote claude -- --dangerously-skip-permissions
```

**Anything else — universal PTY mode:**

```bash
deer-remote pty -- opencode
deer-remote pty --name "openclaude:myapp" -- openclaude
```

The terminal behaves exactly as normal; output is mirrored live to the web
page (ANSI stripped) and web messages are typed into the program's stdin.

`deer-remote` prints the session URL on start. The bridge auto-reconnects
with a buffered queue if the connection drops.

**Resuming/appending to an existing session** — by id (find it via
"Copy session ID" in the web UI's 3-dot menu) or by display name (works for
renamed sessions too — the custom name survives the reconnect):

```bash
deer-remote claude --resume "Renamed session 1" -- --dangerously-skip-permissions
deer-remote claude --resume 45f3e3abc484
```

Equivalent to, and takes priority over, setting `DEER_REMOTE_SESSION=<id>`.

## Endpoints

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /api/remote-control/sessions` | cookie | list sessions |
| `GET /api/remote-control/sessions/{id}/events` | cookie | transcript backlog |
| `PATCH /api/remote-control/sessions/{id}` | cookie | rename / pin / unpin |
| `DELETE /api/remote-control/sessions/{id}` | cookie | delete session + transcript |
| `GET /api/remote-control/resolve-session?name=…` | `REMOTE_CONTROL_TOKEN` | name → id lookup (used by `--resume`) |
| `WS /api/remote-control/ws/agent?token=…` | `REMOTE_CONTROL_TOKEN` | bridge (terminal) side |
| `WS /api/remote-control/ws/client/{id}` | cookie + Origin check | browser side |
