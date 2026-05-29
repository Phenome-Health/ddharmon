import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Route, Switch } from "wouter";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { AppShell } from "@/components/AppShell";
import HomePage from "@/pages/home";
import DashboardPage from "@/pages/dashboard";
import JobsPage from "@/pages/jobs";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <AppShell>
          <Switch>
            <Route path="/" component={HomePage} />
            <Route path="/job/:jobId" component={DashboardPage} />
            <Route path="/jobs" component={JobsPage} />
            <Route>
              <div className="p-8 text-neutral-500">404 — page not found</div>
            </Route>
          </Switch>
        </AppShell>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
