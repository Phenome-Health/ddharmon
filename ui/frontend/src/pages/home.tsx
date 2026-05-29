import { useCallback, useState } from "react";
import { useLocation } from "wouter";
import { useDropzone } from "react-dropzone";
import Papa from "papaparse";
import { toast } from "sonner";
import { Loader2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { detectRoles, startHarmonize } from "@/lib/api";
import { COLUMN_ROLES, type CdeSet, type ClassifyMode, type ColumnRole } from "@/types";

const NONE = "__none__";

interface DictFile {
  file: File;
  cohortName: string;
  headers: string[];
  roles: Record<string, string>; // role -> column (or NONE)
}

function parseHeaders(file: File): Promise<string[]> {
  return new Promise((resolve) => {
    Papa.parse(file, {
      header: true,
      preview: 1,
      skipEmptyLines: true,
      complete: (res) => resolve((res.meta.fields ?? []).filter(Boolean)),
      error: () => resolve([]),
    });
  });
}

export default function HomePage() {
  const [, navigate] = useLocation();
  const [dicts, setDicts] = useState<DictFile[]>([]);
  const [cdeSet, setCdeSet] = useState<CdeSet>("endorsed");
  const [minClusterSize, setMinClusterSize] = useState(15);
  const [classifyMode, setClassifyMode] = useState<ClassifyMode>("none");
  const [displayName, setDisplayName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const onDrop = useCallback(async (accepted: File[]) => {
    const added: DictFile[] = [];
    for (const file of accepted) {
      const headers = await parseHeaders(file);
      let roles: Record<string, string> = {};
      try {
        const detected = await detectRoles(headers);
        roles = detected.columnRoles;
      } catch {
        /* detection best-effort */
      }
      const cohortName = file.name.replace(/\.(csv|tsv|txt)$/i, "");
      added.push({ file, cohortName, headers, roles });
    }
    setDicts((prev) => [...prev, ...added]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"], "text/tab-separated-values": [".tsv"], "text/plain": [".txt"] },
  });

  function setRole(idx: number, role: ColumnRole, column: string) {
    setDicts((prev) => prev.map((d, i) => (i === idx ? { ...d, roles: { ...d.roles, [role]: column } } : d)));
  }
  function setCohort(idx: number, name: string) {
    setDicts((prev) => prev.map((d, i) => (i === idx ? { ...d, cohortName: name } : d)));
  }
  function removeDict(idx: number) {
    setDicts((prev) => prev.filter((_, i) => i !== idx));
  }

  async function run() {
    if (!dicts.length) {
      toast.error("Add at least one data-dictionary file.");
      return;
    }
    for (const d of dicts) {
      const hasKey = ["variable_name", "description", "question_text"].some((r) => d.roles[r] && d.roles[r] !== NONE);
      if (!hasKey) {
        toast.error(`${d.file.name}: map at least variable_name, description, or question_text.`);
        return;
      }
    }
    setSubmitting(true);
    try {
      const config = {
        dictionaries: dicts.map((d) => ({
          filename: d.file.name,
          cohortName: d.cohortName,
          columnRoles: Object.fromEntries(
            Object.entries(d.roles).filter(([, col]) => col && col !== NONE),
          ),
        })),
        cdeSet,
        minClusterSize,
        classifyMode,
        displayName: displayName || undefined,
      };
      const { jobId } = await startHarmonize(
        dicts.map((d) => d.file),
        config,
      );
      navigate(`/job/${jobId}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to start run");
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-ph-ink">New harmonization run</h1>
        <p className="text-sm text-neutral-500">
          Upload cohort data dictionaries, map their columns, and run the clustering → CDE-anchoring pipeline.
        </p>
      </div>

      <div
        {...getRootProps()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed py-12 transition-colors ${
          isDragActive ? "border-ph-navy bg-ph-navy/5" : "border-neutral-300 hover:border-neutral-400"
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="mb-2 h-7 w-7 text-neutral-400" />
        <p className="text-sm font-medium text-neutral-700">Drop CSV/TSV data dictionaries here, or click to browse</p>
        <p className="text-xs text-neutral-400">One file per cohort. CDEs are added automatically server-side.</p>
      </div>

      {dicts.map((d, idx) => (
        <Card key={`${d.file.name}-${idx}`}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              {d.file.name}
              <Badge variant="secondary">{d.headers.length} cols</Badge>
            </CardTitle>
            <Button variant="ghost" size="icon" onClick={() => removeDict(idx)} aria-label="Remove">
              <X className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid max-w-xs gap-1.5">
              <Label>Cohort name</Label>
              <Input value={d.cohortName} onChange={(e) => setCohort(idx, e.target.value)} />
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {COLUMN_ROLES.map((role) => (
                <div key={role} className="grid gap-1.5">
                  <Label className="text-xs">
                    {role}
                    {role === "variable_name" && <span className="text-ph-crimson"> *</span>}
                  </Label>
                  <Select value={d.roles[role] || NONE} onValueChange={(v) => setRole(idx, role, v)}>
                    <SelectTrigger className="h-8">
                      <SelectValue placeholder="—" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NONE}>— none —</SelectItem>
                      {d.headers.map((h) => (
                        <SelectItem key={h} value={h}>
                          {h}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      ))}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Run options</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="grid gap-1.5">
            <Label className="text-xs">CDE catalog</Label>
            <Select value={cdeSet} onValueChange={(v) => setCdeSet(v as CdeSet)}>
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="endorsed">NIH-endorsed (~174)</SelectItem>
                <SelectItem value="full">Full repo (~22.7k)</SelectItem>
                <SelectItem value="none">No CDE</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <Label className="text-xs">Min cluster size</Label>
            <Input
              type="number"
              min={2}
              value={minClusterSize}
              onChange={(e) => setMinClusterSize(Number(e.target.value) || 15)}
              className="h-8"
            />
          </div>
          <div className="grid gap-1.5">
            <Label className="text-xs">Classify (LLM)</Label>
            <Select value={classifyMode} onValueChange={(v) => setClassifyMode(v as ClassifyMode)}>
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None (deterministic only)</SelectItem>
                <SelectItem value="sync">Sync (needs API key)</SelectItem>
                <SelectItem value="batch">Batch (async, hours)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <Label className="text-xs">Run name (optional)</Label>
            <Input value={displayName} onChange={(e) => setDisplayName(e.target.value)} className="h-8" />
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={run} disabled={submitting || !dicts.length} size="lg">
          {submitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Run harmonization
        </Button>
      </div>
    </div>
  );
}
