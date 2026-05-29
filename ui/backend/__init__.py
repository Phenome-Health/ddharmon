"""ddharmon GUI backend — a thin FastAPI app wrapping ddharmon.harmonization.

A background job + in-memory store + SSE progress pattern (no Express/Clerk/Postgres).
Run: ``uvicorn ui.backend.app:app --port 8000``.
"""
