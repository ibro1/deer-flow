"use client";

/**
 * Remote Control — watch and drive terminal coding-agent sessions
 * (Claude Code, opencode, openclaude, ...) bridged into DeerFlow by the
 * `deer-remote` CLI. See `docs/remote-control.md`.
 */

import {
  ChevronLeftIcon,
  CircleIcon,
  SendIcon,
  SquareTerminal,
  WrenchIcon,
} from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";
import {
  fetchRemoteSessions,
  remoteSessionStreamURL,
  type RemoteEvent,
  type RemoteSession,
} from "@/core/remote-control/api";
import { cn } from "@/lib/utils";

const ANSI_RE =
   
  /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|[\x00-\x08\x0b-\x1f]/g;

function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, "");
}

function decodeTty(event: RemoteEvent): string {
  const b64 = (event.data?.b64 as string) ?? "";
  try {
    return stripAnsi(atob(b64));
  } catch {
    return "";
  }
}

/** Group consecutive tty chunks into single terminal blocks for rendering. */
type RenderItem =
  | { kind: "event"; event: RemoteEvent }
  | { kind: "tty"; key: number; text: string };

function groupEvents(events: RemoteEvent[]): RenderItem[] {
  const items: RenderItem[] = [];
  for (const event of events) {
    if (event.type === "tty") {
      const last = items[items.length - 1];
      if (last?.kind === "tty") {
        last.text = (last.text + decodeTty(event)).slice(-60_000);
      } else {
        items.push({ kind: "tty", key: event.seq, text: decodeTty(event) });
      }
    } else {
      items.push({ kind: "event", event });
    }
  }
  return items;
}

export default function RemoteControlPage() {
  return (
    <Suspense fallback={null}>
      <RemoteControlPageInner />
    </Suspense>
  );
}

function RemoteControlPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("session");
  const [sessions, setSessions] = useState<RemoteSession[]>([]);
  const [events, setEvents] = useState<RemoteEvent[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [input, setInput] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const feedRef = useRef<HTMLDivElement | null>(null);

  // Poll the session list.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await fetchRemoteSessions();
        if (!cancelled) setSessions(data);
      } catch {
        // Gateway offline or feature disabled — keep the last list.
      }
    };
    void load();
    const timer = setInterval(() => void load(), 4000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // Live transcript over WebSocket for the selected session.
  useEffect(() => {
    if (!selectedId) return;
    setEvents([]);
    const ws = new WebSocket(remoteSessionStreamURL(selectedId));
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (raw) => {
      try {
        const message = JSON.parse(raw.data as string) as
          | { type: "backlog"; events: RemoteEvent[] }
          | RemoteEvent;
        if ("events" in message && message.type === "backlog") {
          setEvents(message.events);
        } else {
          setEvents((prev) => [...prev, message]);
        }
      } catch {
        // ignore malformed frames
      }
    };
    return () => {
      wsRef.current = null;
      ws.close();
    };
  }, [selectedId]);

  // Auto-scroll on new events.
  useEffect(() => {
    const el = feedRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events]);

  const send = useCallback(() => {
    const text = input.trim();
    const ws = wsRef.current;
    if (!text || ws?.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "user_message", text }));
    setInput("");
  }, [input]);

  const selected = sessions.find((s) => s.id === selectedId);

  const selectSession = useCallback(
    (id: string | null) => {
      router.push(
        id
          ? `/workspace/remote-control?session=${encodeURIComponent(id)}`
          : "/workspace/remote-control",
      );
    },
    [router],
  );

  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch">
        <div className="flex min-h-0 w-full flex-1">
          {/* Mobile-only inline list; on desktop the workspace sidebar
              shows the sessions (RemoteSessionsList). */}
          <SessionList
            className={cn("sm:hidden", selected && "hidden")}
            sessions={sessions}
            selectedId={selectedId}
            onSelect={selectSession}
          />
          <div
            className={cn(
              "min-w-0 flex-1 flex-col",
              selected ? "flex" : "hidden sm:flex",
            )}
          >
            {selected ? (
              <>
                <div className="flex min-w-0 items-center gap-2 border-b px-2 py-2 sm:px-4">
                  <button
                    onClick={() => selectSession(null)}
                    className="hover:bg-accent rounded p-1 sm:hidden"
                    aria-label="Back to sessions"
                  >
                    <ChevronLeftIcon className="size-4" />
                  </button>
                  <CircleIcon
                    className={cn(
                      "size-2.5 shrink-0 fill-current",
                      selected.connected
                        ? "text-green-500"
                        : "text-muted-foreground",
                    )}
                  />
                  <span className="shrink-0 font-medium">{selected.name}</span>
                  <span className="text-muted-foreground hidden min-w-0 truncate text-xs sm:inline">
                    {selected.agent} · {selected.cwd} · {selected.host}
                  </span>
                  {!wsConnected && (
                    <Badge variant="outline" className="ml-auto">
                      reconnecting…
                    </Badge>
                  )}
                </div>
                <div
                  ref={feedRef}
                  className="min-h-0 flex-1 overflow-y-auto px-4 py-4"
                >
                  <div className="mx-auto flex max-w-3xl flex-col gap-3">
                    {groupEvents(events).map((item) =>
                      item.kind === "tty" ? (
                        <TerminalBlock key={`tty-${item.key}`} text={item.text} />
                      ) : (
                        <EventView key={item.event.seq} event={item.event} />
                      ),
                    )}
                  </div>
                </div>
                <Composer
                  value={input}
                  onChange={setInput}
                  onSend={send}
                  disabled={!selected.connected}
                />
              </>
            ) : (
              <EmptyState />
            )}
          </div>
        </div>
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}

function SessionList({
  sessions,
  selectedId,
  onSelect,
  className,
}: {
  sessions: RemoteSession[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex w-full shrink-0 flex-col overflow-y-auto sm:w-72 sm:border-r",
        className,
      )}
    >
      {sessions.length === 0 && (
        <div className="text-muted-foreground p-4 text-sm">
          No sessions yet.
        </div>
      )}
      {sessions.map((session) => (
        <button
          key={session.id}
          onClick={() => onSelect(session.id)}
          className={cn(
            "hover:bg-accent flex flex-col gap-0.5 border-b px-4 py-3 text-left",
            session.id === selectedId && "bg-accent",
          )}
        >
          <span className="flex items-center gap-2 text-sm font-medium">
            <CircleIcon
              className={cn(
                "size-2 fill-current",
                session.connected ? "text-green-500" : "text-muted-foreground",
              )}
            />
            <span className="truncate">{session.name}</span>
          </span>
          <span className="text-muted-foreground truncate text-xs">
            {session.agent} · {session.cwd}
          </span>
          <span className="text-muted-foreground text-xs">
            {new Date(session.last_active * 1000).toLocaleString()}
          </span>
        </button>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="text-muted-foreground flex flex-1 flex-col items-center justify-center gap-3">
      <SquareTerminal className="size-10" />
      <p>Select a session — or start one from any terminal:</p>
      <div className="flex flex-col items-center gap-1 font-mono text-xs">
        <code className="bg-muted rounded px-2 py-1">deer-remote claude</code>
        <code className="bg-muted rounded px-2 py-1">
          deer-remote pty -- opencode
        </code>
      </div>
    </div>
  );
}

function TerminalBlock({ text }: { text: string }) {
  return (
    <pre className="bg-muted/50 max-h-96 overflow-y-auto rounded-lg border p-3 font-mono text-xs whitespace-pre-wrap">
      {text}
    </pre>
  );
}

function Bubble({
  role,
  children,
  highlight,
}: {
  role: string;
  children: React.ReactNode;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="text-muted-foreground mb-1 text-[10px] tracking-wide uppercase">
        {role}
      </div>
      <div
        className={cn(
          "rounded-lg border px-3 py-2 text-sm whitespace-pre-wrap",
          highlight ? "bg-primary/10 border-primary/30" : "bg-card",
        )}
      >
        {children}
      </div>
    </div>
  );
}

function ToolDetails({ title, body }: { title: string; body: string }) {
  return (
    <details className="bg-muted/40 rounded-lg border px-3 py-1.5 text-xs">
      <summary className="flex cursor-pointer items-center gap-1.5 font-mono">
        <WrenchIcon className="size-3" />
        {title}
      </summary>
      <pre className="mt-1 max-h-72 overflow-y-auto font-mono whitespace-pre-wrap">
        {body}
      </pre>
    </details>
  );
}

function EventView({ event }: { event: RemoteEvent }) {
  if (event.type === "status") {
    const state = (event.data?.state as string) ?? "";
    return (
      <div className="text-muted-foreground text-center text-xs">
        — agent {state} —
      </div>
    );
  }
  if (event.type === "remote_user_message") {
    return (
      <Bubble role="you (remote)" highlight>
        {(event.data?.text as string) ?? ""}
      </Bubble>
    );
  }
  if (event.type === "local_user_message") {
    return (
      <Bubble role="terminal user" highlight>
        {(event.data?.text as string) ?? ""}
      </Bubble>
    );
  }
  if (event.type === "error") {
    return (
      <div className="text-destructive text-center text-xs">
        {(event.data?.text as string) ?? "error"}
      </div>
    );
  }
  if (event.type === "agent_event") {
    return <AgentEventView data={event.data ?? {}} />;
  }
  return null;
}

/** Render one Claude Code stream-json event. */
function AgentEventView({ data }: { data: Record<string, unknown> }) {
  const type = data.type as string;
  if (type === "system" && data.subtype === "init") {
    return (
      <div className="text-muted-foreground text-center text-xs">
        session init · {(data.model as string) ?? ""} ·{" "}
        {(data.cwd as string) ?? ""}
      </div>
    );
  }
  if (type === "assistant" || type === "user") {
    const message = data.message as
      | { content?: unknown[] | string }
      | undefined;
    const content = message?.content;
    if (!Array.isArray(content)) return null;
    return (
      <>
        {content.map((raw, index) => {
          const block = raw as Record<string, unknown>;
          if (block.type === "text" && block.text) {
            return (
              <Bubble key={index} role="assistant">
                {block.text as string}
              </Bubble>
            );
          }
          if (block.type === "thinking" && block.thinking) {
            return (
              <ToolDetails
                key={index}
                title="thinking"
                body={block.thinking as string}
              />
            );
          }
          if (block.type === "tool_use") {
            return (
              <ToolDetails
                key={index}
                title={(block.name as string) ?? "tool"}
                body={JSON.stringify(block.input ?? {}, null, 2)}
              />
            );
          }
          if (block.type === "tool_result") {
            const inner = block.content;
            const text =
              typeof inner === "string"
                ? inner
                : Array.isArray(inner)
                  ? inner
                      .map((c) => (c as { text?: string }).text ?? "")
                      .join("\n")
                  : "";
            return (
              <ToolDetails
                key={index}
                title={block.is_error ? "tool result (error)" : "tool result"}
                body={text.slice(0, 8000)}
              />
            );
          }
          return null;
        })}
      </>
    );
  }
  if (type === "result") {
    const cost = (data.total_cost_usd as number) ?? 0;
    const durationMs = (data.duration_ms as number) ?? 0;
    return (
      <div className="text-center font-mono text-xs text-green-600 dark:text-green-500">
        ✔ turn done · ${cost.toFixed(4)} · {Math.round(durationMs / 1000)}s
      </div>
    );
  }
  return null;
}

function Composer({
  value,
  onChange,
  onSend,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled: boolean;
}) {
  return (
    <div className="flex items-end gap-2 border-t p-3">
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSend();
          }
        }}
        placeholder={
          disabled
            ? "Agent offline — reconnect the bridge to send messages"
            : "Send a message into the terminal session… (Enter to send)"
        }
        disabled={disabled}
        className="max-h-40 min-h-11 flex-1 resize-none"
      />
      <Button onClick={onSend} disabled={disabled || !value.trim()} size="icon">
        <SendIcon className="size-4" />
      </Button>
    </div>
  );
}
