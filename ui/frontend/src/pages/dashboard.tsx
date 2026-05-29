import { useMemo, useState } from "react";
import { useParams } from "wouter";
import { toast } from "sonner";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { Ban, Check, Download, Loader2, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useHarmonizeStream } from "@/hooks/use-harmonize-stream";
import { exportUrl, submitVerdict } from "@/lib/api";
import type { Verdict } from "@/types";

const PHASES = ["loading", "embedding", "clustering", "anchoring", "classifying", "complete"];
const VERDICT_STYLES: Record<string, string> = {
  adopt: "bg-success-bg text-success border-success/30",
  refine: "bg-warning-bg text-warning border-warning/30",
  novel: "bg-ph-navy/10 text-ph-navy border-ph-navy/30",
  unaligned: "bg-neutral-100 text-neutral-600 border-neutral-300",
  pending: "bg-white text-neutral-500 border-neutral-300",
};
const BAR_COLORS = ["#005B33", "#B45309", "#113682", "#9CA3AF", "#3AC2CB", "#E21C52"];

function phasePercent(phase: string, completed: number, total: number): number {
  const idx = Math.max(0, PHASES.indexOf(phase));
  if (phase === "classifying" && total > 0) return 80 + (completed / total) * 20;
  if (phase === "complete") return 100;
  return Math.round((idx / (PHASES.length - 1)) * 100);
}

export default function DashboardPage() {
  const { jobId = "" } = useParams();
  const { jobState, error } = useHarmonizeStream(jobId);
  const [decisions, setDecisions] = useState<Record<string, string>>({});
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [filter, setFilter] = useState("all");

  const result = jobState?.result ?? null;
  const verdicts: Verdict[] = useMemo(
    () => (result?.verdicts ?? []).filter((v) => v.mode !== "cde_only" && v.mode !== "noise"),
    [result],
  );
  const filtered = useMemo(
    () => (filter === "all" ? verdicts : verdicts.filter((v) => v.verdict === filter)),
    [verdicts, filter],
  );
  const chartData = useMemo(
    () => Object.entries(result?.summary.counts ?? {}).map(([name, value]) => ({ name, value })),
    [result],
  );

  async function decide(v: Verdict, decision: "approve" | "refine" | "reject") {
    setDecisions((p) => ({ ...p, [v.subClusterId]: decision }));
    try {
      await submitVerdict(jobId, v.subClusterId, decision, notes[v.subClusterId] ?? "");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save decision");
    }
  }

  if (!jobState) {
    return (
      <div className="flex items-center gap-2 p-8 text-neutral-500">
        <Loader2 className="h-4 w-4 animate-spin" /> Connecting to run…
      </div>
    );
  }

  const running = jobState.status !== "complete" && jobState.status !== "error";

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-ph-ink">{jobState.displayName}</h1>
          <p className="text-sm text-neutral-500">Run {jobId.slice(0, 8)}</p>
        </div>
        {result && (
          <div className="flex gap-2">
            <Button variant="outline" size="sm" asChild>
              <a href={exportUrl(jobId, "eitl_tsv")}>
                <Download className="mr-1.5 h-4 w-4" /> EITL TSV
              </a>
            </Button>
            <Button variant="outline" size="sm" asChild>
              <a href={exportUrl(jobId, "decisions_csv")}>
                <Download className="mr-1.5 h-4 w-4" /> Decisions
              </a>
            </Button>
            <Button variant="outline" size="sm" asChild>
              <a href={exportUrl(jobId, "buckets_json")}>
                <Download className="mr-1.5 h-4 w-4" /> Buckets
              </a>
            </Button>
          </div>
        )}
      </div>

      {(running || error) && (
        <Card>
          <CardContent className="space-y-2 py-4">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium capitalize text-neutral-700">
                {error ? "Error" : `Phase: ${jobState.phase}`}
              </span>
              <span className="text-neutral-500">
                {jobState.total > 0 ? `${jobState.completed}/${jobState.total}` : ""}
              </span>
            </div>
            <Progress value={error ? 100 : phasePercent(jobState.phase, jobState.completed, jobState.total)} />
            {error && <p className="text-sm text-danger">{error.message}</p>}
          </CardContent>
        </Card>
      )}

      {result && (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard label="Sub-clusters" value={result.summary.nVerdicts} />
            <StatCard label="CDE-anchored" value={result.summary.nAnchored} />
            <StatCard label="LLM-classified" value={result.summary.nLlmPrompts} />
            <StatCard
              label="Adopt / Refine / Novel"
              value={`${result.summary.counts.adopt ?? 0} / ${result.summary.counts.refine ?? 0} / ${result.summary.counts.novel ?? 0}`}
            />
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Verdict distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12 }} width={28} />
                  <RTooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-base">Review queue ({filtered.length})</CardTitle>
              <Select value={filter} onValueChange={setFilter}>
                <SelectTrigger className="h-8 w-44">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All verdicts</SelectItem>
                  <SelectItem value="adopt">Adopt</SelectItem>
                  <SelectItem value="refine">Refine</SelectItem>
                  <SelectItem value="novel">Novel</SelectItem>
                  <SelectItem value="unaligned">Unaligned</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                </SelectContent>
              </Select>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Sub-cluster</TableHead>
                    <TableHead>Anchor CDE</TableHead>
                    <TableHead>Verdict</TableHead>
                    <TableHead className="text-right">Conf.</TableHead>
                    <TableHead>Cohorts</TableHead>
                    <TableHead>Decision</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.map((v) => (
                    <TableRow key={v.subClusterId}>
                      <TableCell className="max-w-xs">
                        <div className="font-medium text-neutral-800">{v.label || v.subClusterId}</div>
                        <div className="truncate text-xs text-neutral-400">
                          {v.subClusterId} · {v.nFields} fields · {v.evidence}
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-neutral-600">{v.anchorDesignation ?? "—"}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className={VERDICT_STYLES[v.verdict] ?? ""}>
                          {v.verdict}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right text-sm tabular-nums">
                        {v.confidence == null ? "—" : v.confidence.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-xs text-neutral-500">{v.cohorts.join(", ")}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <DecisionBtn active={decisions[v.subClusterId] === "approve"} onClick={() => decide(v, "approve")} title="Approve" color="text-success">
                            <Check className="h-4 w-4" />
                          </DecisionBtn>
                          <DecisionBtn active={decisions[v.subClusterId] === "refine"} onClick={() => decide(v, "refine")} title="Refine" color="text-warning">
                            <Pencil className="h-4 w-4" />
                          </DecisionBtn>
                          <DecisionBtn active={decisions[v.subClusterId] === "reject"} onClick={() => decide(v, "reject")} title="Reject" color="text-danger">
                            <Ban className="h-4 w-4" />
                          </DecisionBtn>
                          <Input
                            placeholder="note"
                            value={notes[v.subClusterId] ?? ""}
                            onChange={(e) => setNotes((p) => ({ ...p, [v.subClusterId]: e.target.value }))}
                            className="h-7 w-28 text-xs"
                          />
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number | string }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="text-xs uppercase tracking-wide text-neutral-400">{label}</div>
        <div className="mt-1 text-2xl font-semibold text-ph-ink">{value}</div>
      </CardContent>
    </Card>
  );
}

function DecisionBtn({
  active,
  onClick,
  title,
  color,
  children,
}: {
  active: boolean;
  onClick: () => void;
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <Button
      type="button"
      variant={active ? "secondary" : "ghost"}
      size="icon"
      className={`h-7 w-7 ${active ? color : "text-neutral-400"}`}
      onClick={onClick}
      title={title}
    >
      {children}
    </Button>
  );
}
