"""ddharmon GUI backend — a thin FastAPI app wrapping ddharmon.harmonization.

Mirrors the biomapper-ui ("Entity Linker Dashboard") python-api pattern
(background job + in-memory store + SSE progress), minus Express/Clerk/Postgres.
Run: ``uvicorn ui.backend.app:app --port 8000``.
"""
