"""Steering: user messages delivered into a run that is already streaming.

The Gateway accepts a steer only for a run live on this worker and puts it in
this process-local inbox, keyed by thread and run. The lead agent's
``SteerMiddleware`` drains the run's entries before each model call and adds
them to the conversation, so the model sees them at its next step and the
thread state (and the UI) records them in order.

A steer that arrives after the run's last model call is never drained; the
inbox drops it with the run, and the client, which did not see the message
appear in the thread, sends it as a new turn instead. Entries of another run
are never delivered: a later run of the same thread has its own ``run_id``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

#: Longest steer accepted, in characters; matches an ordinary chat message.
MAX_STEER_CHARS = 32_000
#: Steers one run may hold at once.
MAX_STEERS_PER_RUN = 20
#: An undrained entry older than this is dropped, bounding memory when a run
#: ends between acceptance and its next model call.
STEER_TTL_SECONDS = 3600.0


@dataclass(frozen=True)
class SteerMessage:
    """One steer: its client id, the user's text, and when it was accepted."""

    steer_id: str
    text: str
    accepted_at: float = field(default_factory=time.monotonic)


class SteerInbox:
    """Process-local steers waiting for their run's next model call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], list[SteerMessage]] = {}

    def put(self, thread_id: str, run_id: str, message: SteerMessage) -> bool:
        """Queue a steer for one run.

        Returns:
            False when the run already holds ``MAX_STEERS_PER_RUN`` steers.
        """
        with self._lock:
            self._expire_locked()
            bucket = self._entries.setdefault((thread_id, run_id), [])
            if len(bucket) >= MAX_STEERS_PER_RUN:
                return False
            bucket.append(message)
            return True

    def drain(self, thread_id: str, run_id: str) -> list[SteerMessage]:
        """Take every steer queued for one run, oldest first."""
        with self._lock:
            return self._entries.pop((thread_id, run_id), [])

    def discard(self, thread_id: str, run_id: str) -> None:
        """Drop a finished run's undelivered steers."""
        with self._lock:
            self._entries.pop((thread_id, run_id), None)

    def _expire_locked(self) -> None:
        cutoff = time.monotonic() - STEER_TTL_SECONDS
        for key in [key for key, bucket in self._entries.items() if all(m.accepted_at < cutoff for m in bucket)]:
            self._entries.pop(key, None)


_inbox = SteerInbox()


def get_steer_inbox() -> SteerInbox:
    """The process's steer inbox, shared by the Gateway and the agent runtime."""
    return _inbox
