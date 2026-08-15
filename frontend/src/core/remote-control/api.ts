import { throwGatewayApiError } from "@/core/api/errors";
import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

export interface RemoteSession {
  id: string;
  name: string;
  agent: string;
  cwd: string;
  host: string;
  created: number;
  last_active: number;
  status: string;
  connected: boolean;
  pinned: boolean;
}

/** One transcript event as stored/broadcast by the gateway relay. */
export interface RemoteEvent {
  seq: number;
  ts: number;
  type:
    | "status"
    | "agent_event"
    | "tty"
    | "remote_user_message"
    | "local_user_message"
    | "error"
    | string;
  data?: Record<string, unknown>;
}

export async function fetchRemoteSessions(): Promise<RemoteSession[]> {
  const response = await fetch(
    `${getBackendBaseURL()}/api/remote-control/sessions`,
  );
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to load remote sessions: ${response.statusText}`,
    );
  }
  return response.json();
}

export async function fetchRemoteSessionEvents(
  sessionId: string,
  after = 0,
): Promise<RemoteEvent[]> {
  const response = await fetch(
    `${getBackendBaseURL()}/api/remote-control/sessions/${encodeURIComponent(sessionId)}/events?after=${after}`,
  );
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to load session events: ${response.statusText}`,
    );
  }
  return response.json();
}

/**
 * Build the WebSocket URL for a live remote-control session.
 *
 * Mirrors ``browserStreamURL``: uses the configured backend base URL when
 * present (split-origin dev/prod), otherwise the current same-origin host.
 */
export function remoteSessionStreamURL(sessionId: string): string {
  const base = getBackendBaseURL();
  const origin =
    base && base.length > 0
      ? base
      : typeof window !== "undefined"
        ? window.location.origin
        : "";
  const wsOrigin = origin.replace(/^http/i, "ws");
  return `${wsOrigin}/api/remote-control/ws/client/${encodeURIComponent(sessionId)}`;
}

export async function updateRemoteSession(
  sessionId: string,
  update: { name?: string; pinned?: boolean },
): Promise<void> {
  const response = await fetch(
    `${getBackendBaseURL()}/api/remote-control/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    },
  );
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to update session: ${response.statusText}`,
    );
  }
}

export async function deleteRemoteSession(sessionId: string): Promise<void> {
  const response = await fetch(
    `${getBackendBaseURL()}/api/remote-control/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    await throwGatewayApiError(
      response,
      `Failed to delete session: ${response.statusText}`,
    );
  }
}
