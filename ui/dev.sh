#!/usr/bin/env bash
# Dev mode: FastAPI backend (:8000, --reload) + Vite dev server (:5173, proxying /api → :8000).
# Open http://localhost:5173
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_PY="$ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python"

echo ">>> starting FastAPI backend on :8000"
"$VENV_PY" -m uvicorn ui.backend.app:app --reload --port 8000 &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT

cd "$ROOT/ui/frontend"
[ -d node_modules ] || npm install
echo ">>> starting Vite dev server on :5173"
API_PROXY_TARGET="http://localhost:8000" npm run dev
