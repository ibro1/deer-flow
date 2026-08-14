"use client";

import { usePathname } from "next/navigation";
import { Suspense } from "react";

import {
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";

import { WorkspaceChannelsList } from "./channels/workspace-channels-list";
import { RecentChatList } from "./recent-chat-list";
import { RemoteSessionsList } from "./remote-sessions-list";
import { WorkspaceHeader } from "./workspace-header";
import { WorkspaceModeTabs } from "./workspace-mode-tabs";
import { WorkspaceNavChatList } from "./workspace-nav-chat-list";
import { WorkspaceNavMenu } from "./workspace-nav-menu";

export function WorkspaceSidebar({
  ...props
}: React.ComponentProps<typeof Sidebar>) {
  const { open: isSidebarOpen } = useSidebar();
  const pathname = usePathname();
  const isRemoteMode = pathname.startsWith("/workspace/remote-control");
  return (
    <>
      <Sidebar variant="sidebar" collapsible="icon" {...props}>
        <SidebarHeader className="py-0">
          <WorkspaceHeader />
          <WorkspaceModeTabs />
        </SidebarHeader>
        <SidebarContent>
          {isRemoteMode ? (
            <Suspense fallback={null}>
              {isSidebarOpen && <RemoteSessionsList />}
            </Suspense>
          ) : (
            <>
              <WorkspaceNavChatList />
              <WorkspaceChannelsList />
              {isSidebarOpen && <RecentChatList />}
            </>
          )}
        </SidebarContent>
        <SidebarFooter>
          <WorkspaceNavMenu />
        </SidebarFooter>
        <SidebarRail />
      </Sidebar>
    </>
  );
}
