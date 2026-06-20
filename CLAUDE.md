# CLAUDE.md — ddharmon

ddharmon (Data Dictionary Harmonization Tool) clusters equivalent variables across
biomedical data dictionaries and anchors each to a Common Data Element (CDE), routing every
recommendation to expert review. Full pipeline in `README.md`; algorithm + lineage in
`docs/v1_methods.md`.

## Repo topology (important)

- **Canonical + publisher:** `Phenome-Health/ddharmon` — all development *and* PyPI releases
  happen here.
- **Mirror:** `trentleslie/ddharmon` — content mirror for Greptile reviews; it **never
  publishes**. Do not cut releases or push release tags from the mirror.

## Environment & setup

- Python **3.12+**, managed with **uv**.
- `uv sync --extra all` for the full pipeline (`--extra dev` adds test/lint tooling).
- Copy `.env.example` → `.env`; set `ANTHROPIC_API_KEY` for the classify pass (sync/batch LLM).
- The NIH CDE catalog is **not bundled** — flatten it locally with
  `scripts/flatten_cde_repo.py <All-CDEs.json> <out.tsv>`. Without it the pipeline still
  clusters cohort variables (`cdeSet = none`).

## Common commands

- `./scripts/check.sh` — lint + format + typecheck + test (run before pushing)
- `./scripts/fix.sh` — auto-fix lint + format
- `pytest` — tests; integration tests (`-m integration`) need API keys and may incur cost

## Conventions

- **ruff** + **black**, line length **120**; **pyright** in basic mode (`notebooks/`,
  `tests/` excluded from type-checking).
- Source lives under `src/ddharmon/` — see the README "Project structure" map. Data models
  are plain dataclasses.
- Entity resolution is intentionally an *external service*, not a dependency — keep ddharmon
  focused on data-dictionary harmonization.

## Releasing to PyPI

Use the **`publish-to-pypi` skill** → [`.claude/skills/publish-to-pypi/SKILL.md`](.claude/skills/publish-to-pypi/SKILL.md).
It covers bumping the `pyproject.toml` version + `CHANGELOG.md`, tagging, and publishing to
<https://pypi.org/project/ddharmon/> via **GitHub Release → Trusted Publishing (OIDC, no
token)**. Triggers: "release ddharmon", "publish to pypi", "cut a release". Releases run from
`Phenome-Health/ddharmon` only.

## Workflow

- All changes go through **pull requests** — no direct pushes to environments.
