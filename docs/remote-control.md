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
`REMOTE_CONTROL_DB` (default `.deerflow/remote_control.db`).

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
deer-remote claude -- --permission-mode acceptEdits --model sonnet
```

Uses `claude -p --input-format stream-json --output-format stream-json
--verbose`, the officially supported bidirectional streaming interface.
Type in the terminal *or* from the web page — both are injected as user
turns and both sides see everything.

**Anything else — universal PTY mode:**

```bash
deer-remote pty -- opencode
deer-remote pty --name "openclaude:myapp" -- openclaude
```

The terminal behaves exactly as normal; output is mirrored live to the web
page (ANSI stripped) and web messages are typed into the program's stdin.

`deer-remote` prints the session URL on start. Set
`DEER_REMOTE_SESSION=<id>` to resume/append to an existing session id. The
bridge auto-reconnects with a buffered queue if the connection drops.

## Endpoints

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /api/remote-control/sessions` | cookie | list sessions |
| `GET /api/remote-control/sessions/{id}/events` | cookie | transcript backlog |
| `WS /api/remote-control/ws/agent?token=…` | `REMOTE_CONTROL_TOKEN` | bridge (terminal) side |
| `WS /api/remote-control/ws/client/{id}` | cookie + Origin check | browser side |
