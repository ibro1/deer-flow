"use client";

/**
 * Sidebar content for Remote mode: the list of terminal coding-agent
 * sessions (recent, active, inactive). Replaces the chat nav while the
 * user is on /workspace/remote-control, mirroring how Recent chats
 * serves Chat mode. Selecting a session navigates to
 * /workspace/remote-control?session=<id>.
 */

import { CircleIcon } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useI18n } from "@/core/i18n/hooks";
import {
  fetchRemoteSessions,
  type RemoteSession,
} from "@/core/remote-control/api";
import { cn } from "@/lib/utils";

export function RemoteSessionsList() {
  const { t } = useI18n();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("session");
  const [sessions, setSessions] = useState<RemoteSession[]>([]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await fetchRemoteSessions();
        if (!cancelled) setSessions(data);
      } catch {
        // Gateway offline — keep the last list.
      }
    };
    void load();
    const timer = setInterval(() => void load(), 4000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <SidebarGroup>
      <SidebarGroupLabel>{t.sidebar.remoteSessions}</SidebarGroupLabel>
      <SidebarMenu>
        {sessions.length === 0 && (
          <div className="text-muted-foreground px-2 py-1 text-xs">
            {t.sidebar.remoteSessionsEmpty}
          </div>
        )}
        {sessions.map((session) => (
          <SidebarMenuItem key={session.id}>
            <SidebarMenuButton
              isActive={session.id === selectedId}
              className="h-auto"
              asChild
            >
              <Link
                href={`/workspace/remote-control?session=${encodeURIComponent(session.id)}`}
                className="flex w-full flex-col items-start gap-0.5"
              >
                <span className="flex w-full min-w-0 items-center gap-2">
                  <CircleIcon
                    className={cn(
                      "size-2 shrink-0 fill-current",
                      session.connected
                        ? "text-green-500"
                        : "text-muted-foreground/50",
                    )}
                  />
                  <span className="truncate text-sm font-medium">
                    {session.name}
                  </span>
                </span>
                <span className="text-muted-foreground w-full truncate pl-4 text-xs">
                  {session.agent} · {session.cwd}
                </span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        ))}
      </SidebarMenu>
    </SidebarGroup>
  );
}
