// Mirrors the camelCase shapes emitted by ui/backend (jobs.py / runner.py).

export type JobStatus =
  | "pending"
  | "loading"
  | "embedding"
  | "clustering"
  | "anchoring"
  | "classifying"
  | "complete"
  | "error";

export interface Verdict {
  subClusterId: string;
  parentTopicId: number;
  subLabel: number;
  mode: string; // harmonize | kg_only | single_cohort | cde_only | noise
  verdict: string; // adopt | refine | novel | unaligned | pending
  parentCdeId: string | null;
  confidence: number | null;
  evidence: string;
  label: string;
  cohorts: string[];
  nFields: number;
  encodedFraction: number;
  anchorDesignation: string | null;
  decidedBy: string;
}

export interface ResultSummary {
  nVerdicts: number;
  nLlmPrompts: number;
  nAnchored: number;
  counts: Record<string, number>;
}

export interface HarmonizationResult {
  verdicts: Verdict[];
  summary: ResultSummary;
}

export interface JobResult {
  jobId: string;
  displayName: string;
  status: JobStatus;
  phase: string;
  completed: number;
  total: number;
  errorMessage: string | null;
  result: HarmonizationResult | null;
  config: Record<string, unknown>;
  decisions: Record<string, { decision: string; note: string }>;
  createdAt: number;
  updatedAt: number;
}

export interface JobSummary extends Omit<JobResult, "result"> {
  nVerdicts: number;
}

export type ClassifyMode = "none" | "sync" | "batch";
export type CdeSet = "endorsed" | "full" | "none";

export interface DictSpec {
  filename: string;
  cohortName: string;
  columnRoles: Record<string, string>;
}

export interface RunConfig {
  dictionaries: DictSpec[];
  cdeSet: CdeSet;
  minClusterSize: number;
  classifyMode: ClassifyMode;
  displayName?: string;
}

// load_dictionary column roles the UI lets you map (value = source column).
export const COLUMN_ROLES = [
  "variable_name",
  "description",
  "question_text",
  "value_encoding",
  "data_type",
  "units",
  "category",
  "field_id",
  "standard_code",
] as const;
export type ColumnRole = (typeof COLUMN_ROLES)[number];
