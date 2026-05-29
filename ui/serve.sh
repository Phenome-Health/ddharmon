#!/usr/bin/env bash
# Production-ish mode: build the frontend, then serve it + the API from one FastAPI process.
# Open http://localhost:8000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_PY="$ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python"

cd "$ROOT/ui/frontend"
[ -d node_modules ] || npm install
echo ">>> building frontend"
npm run build

cd "$ROOT"
echo ">>> serving app + API on :8000"
exec "$VENV_PY" -m uvicorn ui.backend.app:app --port 8000
