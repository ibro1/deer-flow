"use client";

/**
 * Desktop-only top-level mode switcher: Chat vs Remote.
 *
 * Chat is the normal DeerFlow experience (chats, agents, scheduled tasks);
 * Remote is the terminal coding-agent sessions view (/workspace/remote-control).
 * On mobile this is hidden — the sidebar sheet keeps its "Remote control"
 * menu item instead.
 */

import { MessagesSquare, SquareTerminal } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useSidebar } from "@/components/ui/sidebar";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

export function WorkspaceModeTabs() {
  const { t } = useI18n();
  const { state } = useSidebar();
  const pathname = usePathname();
  const isRemote = pathname.startsWith("/workspace/remote-control");

  if (state === "collapsed") {
    return null;
  }

  return (
    <div className="bg-muted mx-2 mb-1 hidden grid-cols-2 gap-1 rounded-lg p-1 md:grid">
      <Link
        href="/workspace/chats"
        className={cn(
          "flex items-center justify-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition-colors",
          !isRemote
            ? "bg-background text-foreground shadow-sm"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <MessagesSquare className="size-3.5" />
        <span>{t.sidebar.tabs.chat}</span>
      </Link>
      <Link
        href="/workspace/remote-control"
        className={cn(
          "flex items-center justify-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition-colors",
          isRemote
            ? "bg-background text-foreground shadow-sm"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <SquareTerminal className="size-3.5" />
        <span>{t.sidebar.tabs.remote}</span>
      </Link>
    </div>
  );
}
