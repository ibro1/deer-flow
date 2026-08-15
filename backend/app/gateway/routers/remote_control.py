"""Remote Control — bridge terminal coding agents into the DeerFlow UI.

The same idea as Claude Code's ``/remote-control``, self-hosted and
agent-agnostic. A thin ``deer-remote`` bridge CLI (see ``scripts/deer-remote``)
runs next to any terminal coding agent (Claude Code in stream-json mode, or
anything under a PTY: opencode, openclaude, aider, ...) and connects to
``/api/remote-control/ws/agent``, streaming session events up and receiving
user messages to inject back into the terminal session.

Browsers (the ``/workspace/remote-control`` page) connect to
``/api/remote-control/ws/client/{session_id}`` to watch the transcript live
and send messages into the session.

Auth model:
  * Browser REST + WS — normal DeerFlow cookie auth (WS replicates the
    cookie resolution like ``browser.py`` does, since BaseHTTPMiddleware does
    not see WebSocket upgrades) plus Origin allow-listing.
  * Agent WS — bridges are headless, non-browser clients with no cookies.
    They authenticate with the shared secret ``REMOTE_CONTROL_TOKEN`` env
    var. If the env var is unset the agent endpoint is disabled (403).

Persistence: a dedicated SQLite file (transcripts are an append-only event
log, deliberately independent of the main persistence engine). Path comes
from ``REMOTE_CONTROL_DB`` (default: ``.deerflow/remote_control.db`` under
the working directory).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/remote-control", tags=["remote-control"])

MAX_BACKLOG_SEND = 2000  # events replayed to a freshly-connected client
MAX_EVENT_BYTES = 512 * 1024  # reject absurdly large single events


# ---------------------------------------------------------------------------
# Persistence (dedicated SQLite event log)
# ---------------------------------------------------------------------------


def _db_path() -> str:
    return os.environ.get(
        "REMOTE_CONTROL_DB",
        str(Path(".deerflow") / "remote_control.db"),
    )


def _db() -> sqlite3.Connection:
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rc_sessions (
            id TEXT PRIMARY KEY,
            name TEXT, agent TEXT, cwd TEXT, host TEXT,
            created REAL, last_active REAL,
            status TEXT DEFAULT 'live'
        );
        CREATE TABLE IF NOT EXISTS rc_events (
            session_id TEXT, seq INTEGER, ts REAL, payload TEXT,
            PRIMARY KEY (session_id, seq)
        );
        """
    )
    # Lightweight migrations for pre-existing databases.
    for ddl in (
        "ALTER TABLE rc_sessions ADD COLUMN custom_name TEXT",
        "ALTER TABLE rc_sessions ADD COLUMN pinned INTEGER DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


# ---------------------------------------------------------------------------
# In-memory session hub
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self, sid: str):
        self.id = sid
        self.agent_ws: Optional[WebSocket] = None
        self.client_wss: Set[WebSocket] = set()
        self.seq = 0
        self.lock = asyncio.Lock()


_sessions: Dict[str, _Session] = {}


def _get_session(sid: str) -> _Session:
    if sid not in _sessions:
        sess = _Session(sid)
        conn = _db()
        row = conn.execute(
            "SELECT MAX(seq) AS m FROM rc_events WHERE session_id=?", (sid,)
        ).fetchone()
        conn.close()
        sess.seq = (row["m"] or 0) if row else 0
        _sessions[sid] = sess
    return _sessions[sid]


async def _record_and_broadcast(sess: _Session, payload: dict) -> None:
    """Persist an event and fan it out to connected browser clients."""
    async with sess.lock:
        sess.seq += 1
        payload["seq"] = sess.seq
        payload.setdefault("ts", time.time())

        def _persist() -> None:
            conn = _db()
            conn.execute(
                "INSERT OR REPLACE INTO rc_events VALUES (?,?,?,?)",
                (sess.id, sess.seq, payload["ts"], json.dumps(payload)),
            )
            conn.execute(
                "UPDATE rc_sessions SET last_active=? WHERE id=?",
                (time.time(), sess.id),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_persist)

    text = json.dumps(payload)
    dead = []
    for cws in list(sess.client_wss):
        try:
            await cws.send_text(text)
        except Exception:
            dead.append(cws)
    for d in dead:
        sess.client_wss.discard(d)


def _set_status(sid: str, status: str) -> None:
    conn = _db()
    conn.execute(
        "UPDATE rc_sessions SET status=?, last_active=? WHERE id=?",
        (status, time.time(), sid),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _agent_token() -> str:
    return os.environ.get("REMOTE_CONTROL_TOKEN", "")


def _agent_token_ok(supplied: str) -> bool:
    expected = _agent_token()
    if not expected:
        return False  # feature disabled for agents until a token is configured
    return secrets.compare_digest(supplied or "", expected)


async def _require_user(request: Request) -> Any:
    from app.gateway.deps import get_optional_user_from_request

    user = await get_optional_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ---------------------------------------------------------------------------
# REST — session list + transcript backlog
# ---------------------------------------------------------------------------


@router.get("/sessions")
async def list_sessions(request: Request) -> list[dict[str, Any]]:
    await _require_user(request)

    def _query() -> list[dict[str, Any]]:
        conn = _db()
        rows = conn.execute(
            # Stable ordering: pinned first, then creation time. Ordering by
            # last_active makes rows jump around on every poll while agents
            # are streaming (activity is signalled by the live dot instead).
            "SELECT * FROM rc_sessions ORDER BY pinned DESC, created DESC LIMIT 200"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    out = await asyncio.to_thread(_query)
    for d in out:
        sess = _sessions.get(d["id"])
        d["connected"] = bool(sess and sess.agent_ws is not None)
        d["name"] = d.get("custom_name") or d["name"]
        d["pinned"] = bool(d.get("pinned"))
    return out


@router.get("/sessions/{sid}/events")
async def session_events(
    sid: str, request: Request, after: int = Query(0)
) -> list[dict[str, Any]]:
    await _require_user(request)

    def _query() -> list[dict[str, Any]]:
        conn = _db()
        rows = conn.execute(
            "SELECT payload FROM rc_events WHERE session_id=? AND seq>? "
            "ORDER BY seq LIMIT 5000",
            (sid, after),
        ).fetchall()
        conn.close()
        return [json.loads(r["payload"]) for r in rows]

    return await asyncio.to_thread(_query)


class SessionUpdateRequest(BaseModel):
    name: str | None = None
    pinned: bool | None = None


@router.patch("/sessions/{sid}")
async def update_session(
    sid: str, body: SessionUpdateRequest, request: Request
) -> dict[str, Any]:
    """Rename (stored as custom_name so bridge reconnects don't clobber it)
    and/or pin a session."""
    await _require_user(request)

    def _update() -> int:
        conn = _db()
        sets, params = [], []
        if body.name is not None:
            sets.append("custom_name=?")
            params.append(body.name.strip()[:200])
        if body.pinned is not None:
            sets.append("pinned=?")
            params.append(1 if body.pinned else 0)
        if not sets:
            conn.close()
            return 1
        params.append(sid)
        cur = conn.execute(
            f"UPDATE rc_sessions SET {', '.join(sets)} WHERE id=?", params
        )
        conn.commit()
        conn.close()
        return cur.rowcount

    if await asyncio.to_thread(_update) == 0:
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str, request: Request) -> dict[str, Any]:
    """Delete a session and its transcript. Live agent bridges are
    disconnected first (they may re-register as a fresh session)."""
    await _require_user(request)
    sess = _sessions.pop(sid, None)
    if sess is not None and sess.agent_ws is not None:
        try:
            await sess.agent_ws.close(code=4410)
        except Exception:
            pass

    def _delete() -> int:
        conn = _db()
        conn.execute("DELETE FROM rc_events WHERE session_id=?", (sid,))
        cur = conn.execute("DELETE FROM rc_sessions WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        return cur.rowcount

    if await asyncio.to_thread(_delete) == 0:
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket — agent (bridge) side
# ---------------------------------------------------------------------------


@router.websocket("/ws/agent")
async def ws_agent(ws: WebSocket) -> None:
    token = ws.query_params.get("token", "")
    if not _agent_token_ok(token):
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        hello = json.loads(await ws.receive_text())
    except Exception:
        await ws.close(code=4400)
        return
    if hello.get("type") != "hello":
        await ws.close(code=4400)
        return

    sid = hello.get("session_id") or secrets.token_hex(6)
    sess = _get_session(sid)

    def _upsert() -> None:
        conn = _db()
        conn.execute(
            """INSERT INTO rc_sessions
                 (id, name, agent, cwd, host, created, last_active, status)
               VALUES (?,?,?,?,?,?,?, 'live')
               ON CONFLICT(id) DO UPDATE SET
                 status='live', last_active=excluded.last_active,
                 name=excluded.name, agent=excluded.agent,
                 cwd=excluded.cwd, host=excluded.host""",
            (
                sid,
                str(hello.get("name") or sid)[:200],
                str(hello.get("agent") or "unknown")[:100],
                str(hello.get("cwd") or "")[:500],
                str(hello.get("host") or "")[:200],
                time.time(),
                time.time(),
            ),
        )
        conn.commit()
        conn.close()

    await asyncio.to_thread(_upsert)
    sess.agent_ws = ws
    await ws.send_text(json.dumps({"type": "ready", "session_id": sid}))
    await _record_and_broadcast(
        sess, {"type": "status", "data": {"state": "connected", "meta": hello}}
    )
    logger.info("remote-control agent connected: session=%s agent=%s", sid, hello.get("agent"))
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > MAX_EVENT_BYTES:
                continue
            msg = json.loads(raw)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue
            await _record_and_broadcast(sess, msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("remote-control agent ws error: session=%s", sid)
    finally:
        if sess.agent_ws is ws:
            sess.agent_ws = None
            await asyncio.to_thread(_set_status, sid, "offline")
            await _record_and_broadcast(
                sess, {"type": "status", "data": {"state": "disconnected"}}
            )
            logger.info("remote-control agent disconnected: session=%s", sid)


# ---------------------------------------------------------------------------
# WebSocket — browser (client) side
# ---------------------------------------------------------------------------


@router.websocket("/ws/client/{sid}")
async def ws_client(ws: WebSocket, sid: str) -> None:
    # Reuse the browser-stream auth helpers: cookie -> user resolution and
    # Origin allow-listing (WS upgrades bypass the HTTP middlewares).
    from app.gateway.routers.browser import _authenticate_ws, _ws_origin_allowed

    if not _ws_origin_allowed(ws):
        await ws.close(code=4403)
        return
    user = await _authenticate_ws(ws)
    if user is None:
        await ws.close(code=4401)
        return

    await ws.accept()
    sess = _get_session(sid)
    sess.client_wss.add(ws)

    def _backlog() -> list[dict[str, Any]]:
        conn = _db()
        rows = conn.execute(
            "SELECT payload FROM rc_events WHERE session_id=? "
            "ORDER BY seq DESC LIMIT ?",
            (sid, MAX_BACKLOG_SEND),
        ).fetchall()
        conn.close()
        return [json.loads(r["payload"]) for r in reversed(rows)]

    backlog = await asyncio.to_thread(_backlog)
    await ws.send_text(json.dumps({"type": "backlog", "events": backlog}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") != "user_message":
                continue
            text = str(msg.get("text", ""))[:100_000]
            if sess.agent_ws is not None:
                await sess.agent_ws.send_text(
                    json.dumps({"type": "user_message", "text": text})
                )
                await _record_and_broadcast(
                    sess, {"type": "remote_user_message", "data": {"text": text}}
                )
            else:
                await ws.send_text(
                    json.dumps({"type": "error", "data": {"text": "agent offline"}})
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("remote-control client ws error: session=%s", sid)
    finally:
        sess.client_wss.discard(ws)
