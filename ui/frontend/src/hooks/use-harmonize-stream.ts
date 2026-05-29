// SSE hook for live harmonization-run progress.
// Adapted from biomapper-ui's use-mapping-stream.ts (same EventSource + exponential-backoff
// retry + terminal-status close), pointed at /api/harmonize/stream and typed to JobResult.
import { useEffect, useRef, useState } from "react";
import type { JobResult } from "@/types";

const MAX_RETRIES = 5;
const BASE_RETRY_MS = 1500;

export interface StreamError {
  message: string;
}

export function useHarmonizeStream(jobId: string, enabled = true) {
  const [jobState, setJobState] = useState<JobResult | null>(null);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<StreamError | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    if (!enabled || !jobId) return;

    function connect(retryCount: number) {
      if (!mountedRef.current) return;
      const es = new EventSource(`/api/harmonize/stream/${jobId}`);
      esRef.current = es;

      es.addEventListener("progress", (e) => {
        if (!mountedRef.current) return;
        try {
          const data: JobResult = JSON.parse((e as MessageEvent).data);
          setJobState(data);
          if (data.status === "complete") {
            setDone(true);
            es.close();
          } else if (data.status === "error") {
            setError({ message: data.errorMessage ?? "Harmonization failed" });
            setDone(true);
            es.close();
          }
        } catch (err) {
          console.error("[SSE] error parsing progress payload", err);
        }
      });

      es.onerror = () => {
        es.close();
        if (!mountedRef.current) return;
        if (retryCount < MAX_RETRIES) {
          const delay = BASE_RETRY_MS * Math.pow(2, retryCount);
          retryTimerRef.current = setTimeout(() => connect(retryCount + 1), delay);
        } else {
          setError({ message: "Connection to harmonization service lost after multiple retries" });
          setDone(true);
        }
      };
    }

    connect(0);
    return () => {
      mountedRef.current = false;
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      esRef.current?.close();
    };
  }, [jobId, enabled]);

  return { jobState, done, error };
}
