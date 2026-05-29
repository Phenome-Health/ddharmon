import type { ReactNode } from "react";
import { Link, useLocation } from "wouter";
import { Network, Plus, ListChecks } from "lucide-react";
import { cn } from "@/lib/utils";

function NavLink({ href, icon, label }: { href: string; icon: ReactNode; label: string }) {
  const [loc] = useLocation();
  const active = loc === href || (href !== "/" && loc.startsWith(href));
  return (
    <Link
      href={href}
      className={cn(
        "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
        active ? "bg-ph-navy text-white" : "text-neutral-600 hover:bg-neutral-100",
      )}
    >
      {icon}
      {label}
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col bg-neutral-50">
      <header className="flex items-center justify-between border-b border-neutral-200 bg-white px-6 py-3">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-ph-navy text-white">
            <Network className="h-4 w-4" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-ph-ink">ddharmon</div>
            <div className="text-[11px] text-neutral-500">Sub-cluster-anchored CDE harmonization</div>
          </div>
        </Link>
        <nav className="flex items-center gap-1">
          <NavLink href="/" icon={<Plus className="h-4 w-4" />} label="New run" />
          <NavLink href="/jobs" icon={<ListChecks className="h-4 w-4" />} label="Runs" />
        </nav>
      </header>
      <main className="mx-auto w-full max-w-screen-2xl flex-1 px-6 py-6">{children}</main>
    </div>
  );
}
