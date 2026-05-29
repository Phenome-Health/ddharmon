# ddharmon GUI

A simple web GUI for the ddharmon v1 harmonization pipeline, modeled on the
[biomapper-ui](https://github.com/trentleslie/biomapper-ui) ("Entity Linker Dashboard"):
React + Vite + Tailwind + shadcn/ui frontend talking to a FastAPI backend that wraps
`ddharmon.harmonization`.

**Workflow:** upload cohort data dictionaries → map columns → run the pipeline (cluster →
value sub-cluster → CDE anchor → adopt/refine/novel) with live progress → review the
recommendations (approve / refine / reject) → export the EITL queue.

```
React+Vite (ui/frontend, :5173)  →  /api proxy  →  FastAPI (ui/backend, :8000)  →  ddharmon
```

## Prerequisites

- Python env with the `ui` extra: `uv sync --extra dev` (or `pip install -e ".[ui]"`)
- Node 20+ and npm (or pnpm)
- CDE catalog flattened locally (for `cdeSet` = endorsed/full):
  `python scripts/flatten_cde_repo.py data/examples/cde/All-CDEs.json data/examples/cde/all_cdes_flat.tsv`
- `ANTHROPIC_API_KEY` in the environment — only needed for `classifyMode` = `sync`/`batch`
  (the default `none` runs the full clustering + anchoring with no LLM and no key)

## Run

**Dev** (hot reload; two processes):

```bash
./ui/dev.sh          # backend :8000 + Vite :5173 → open http://localhost:5173
```

**Serve** (build once; single FastAPI process serves the SPA + API):

```bash
./ui/serve.sh        # → open http://localhost:8000
```

## Notes

- Jobs are kept in-memory — they're lost when the backend restarts (fine for single-user v1).
- `classifyMode=none` shows CDE-anchored sub-clusters as `pending` (un-classified); choose
  `sync` (inline, needs API key) or `batch` (Anthropic Batch API, async) to get adopt/refine/novel.
- Uploaded files + batch artifacts land in `.ddharmon_ui/<jobId>/` (gitignored).
