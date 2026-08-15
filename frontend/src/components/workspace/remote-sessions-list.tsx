"use client";

/**
 * Sidebar content for Remote mode: the list of terminal coding-agent
 * sessions (pinned first, then newest). Each row has a 3-dot menu with
 * pin/unpin, rename, copy-id and delete. Selecting a session navigates
 * to /workspace/remote-control?session=<id>.
 */

import {
  CircleIcon,
  CopyIcon,
  MoreHorizontal,
  PencilIcon,
  PinIcon,
  PinOffIcon,
  Trash2Icon,
} from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useI18n } from "@/core/i18n/hooks";
import {
  deleteRemoteSession,
  fetchRemoteSessions,
  updateRemoteSession,
  type RemoteSession,
} from "@/core/remote-control/api";
import { cn } from "@/lib/utils";

export function RemoteSessionsList() {
  const { t } = useI18n();
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("session");
  const [sessions, setSessions] = useState<RemoteSession[]>([]);
  const [renaming, setRenaming] = useState<RemoteSession | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleting, setDeleting] = useState<RemoteSession | null>(null);

  const load = useCallback(async () => {
    try {
      setSessions(await fetchRemoteSessions());
    } catch {
      // Gateway offline — keep the last list.
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 4000);
    return () => clearInterval(timer);
  }, [load]);

  const togglePin = useCallback(
    async (session: RemoteSession) => {
      try {
        await updateRemoteSession(session.id, { pinned: !session.pinned });
        await load();
      } catch {
        // surfaced by the next poll
      }
    },
    [load],
  );

  const submitRename = useCallback(async () => {
    if (!renaming) return;
    const name = renameValue.trim();
    if (name && name !== renaming.name) {
      try {
        await updateRemoteSession(renaming.id, { name });
        await load();
      } catch {
        // surfaced by the next poll
      }
    }
    setRenaming(null);
  }, [renaming, renameValue, load]);

  const confirmDelete = useCallback(async () => {
    if (!deleting) return;
    try {
      await deleteRemoteSession(deleting.id);
      if (selectedId === deleting.id) {
        router.push("/workspace/remote-control");
      }
      await load();
    } catch {
      // surfaced by the next poll
    }
    setDeleting(null);
  }, [deleting, selectedId, router, load]);

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
                <span className="flex w-full min-w-0 items-center gap-2 pr-5">
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
                  {session.pinned && (
                    <PinIcon className="text-muted-foreground size-3 shrink-0" />
                  )}
                </span>
                <span className="text-muted-foreground w-full truncate pl-4 text-xs">
                  {session.agent} · {session.cwd}
                </span>
              </Link>
            </SidebarMenuButton>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuAction showOnHover aria-label={t.common.more}>
                  <MoreHorizontal />
                </SidebarMenuAction>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="right" align="start" className="w-48">
                <DropdownMenuItem onClick={() => void togglePin(session)}>
                  {session.pinned ? (
                    <PinOffIcon className="size-4" />
                  ) : (
                    <PinIcon className="size-4" />
                  )}
                  {session.pinned
                    ? t.sidebar.remoteSessionUnpin
                    : t.sidebar.remoteSessionPin}
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => {
                    setRenameValue(session.name);
                    setRenaming(session);
                  }}
                >
                  <PencilIcon className="size-4" />
                  {t.common.rename}
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => void navigator.clipboard.writeText(session.id)}
                >
                  <CopyIcon className="size-4" />
                  {t.sidebar.remoteSessionCopyId}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  variant="destructive"
                  onClick={() => setDeleting(session)}
                >
                  <Trash2Icon className="size-4" />
                  {t.common.delete}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        ))}
      </SidebarMenu>

      {/* Rename dialog */}
      <Dialog
        open={renaming !== null}
        onOpenChange={(open) => !open && setRenaming(null)}
      >
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>{t.sidebar.remoteSessionRenameTitle}</DialogTitle>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void submitRename();
              }
            }}
            autoFocus
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenaming(null)}>
              {t.common.cancel}
            </Button>
            <Button onClick={() => void submitRename()}>{t.common.save}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleting !== null}
        onOpenChange={(open) => !open && setDeleting(null)}
      >
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>{t.sidebar.remoteSessionDeleteConfirm}</DialogTitle>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleting(null)}>
              {t.common.cancel}
            </Button>
            <Button variant="destructive" onClick={() => void confirmDelete()}>
              {t.common.delete}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </SidebarGroup>
  );
}
