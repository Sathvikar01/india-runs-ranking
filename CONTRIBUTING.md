# Contributing

Thank you for your interest in the **India Runs Data & AI Challenge — Candidate Intelligence Platform**!

This project is a competition submission, but we want it to stay useful and reproducible beyond the competition. Contributions that improve evaluation rigor, ranking quality, or documentation are welcome.

## Development setup

```bash
git clone https://github.com/arsat/india-runs-ranking.git
cd india-runs-ranking
python -m venv .venv
. .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Branch from `main`: `git switch -c feat/<short-description>`
2. Make small, focused commits. Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat: …` new capability
   - `fix: …` bug fix
   - `refactor: …` no-behavior-change cleanup
   - `docs: …` documentation only
   - `test: …` add/adjust tests
   - `chore: …` tooling / CI / deps
3. Run the local quality gate before pushing:
   ```bash
   ruff check src tests
   black --check src tests
   mypy src
   pytest -q
   ```
4. Push and open a Pull Request. CI must pass.
5. The PR description should explain **what** changed and **why**, with evidence (metric, screenshot, ablation).

## Code style

- Python 3.11+. Strict type hints in `src/`.
- Keep public APIs narrow. Prefer pure functions; isolate side effects in `serving/`.
- Every new module gets a test. Coverage on `src/` should stay ≥ 90 %.

## Architecture

- `src/ingestion` parses raw JSONL.
- `src/preprocessing` derives the `deep_profile` text and features.
- `src/retrieval` builds BM25 + dense indexes (offline build only).
- `src/ranking` runs cross-encoder and LTR.
- `src/behavioral` computes availability, honeypot-risk, JD-penalties.
- `src/feature_store` is the parquet-backed artifact layer.
- `src/evaluation` owns metrics and ablation runners.
- `src/serving` is the entry point that the sandbox reproduces (≤ 5 min, 16 GB, CPU only, no network).
- `src/training` owns hard-negative mining and LTR training.

Anything that touches the network (LLM calls, embedding downloads) belongs in `scripts/build_*` or `src/training`, **never** in `src/serving/`.

## Reporting bugs

Use the [Bug Report](../../.github/ISSUE_TEMPLATE/bug_report.md) template.

## Suggesting features

Use the [Feature Request](../../.github/ISSUE_TEMPLATE/feature_request.md) template.

## Code of Conduct

Be respectful. We judge ideas, not people.
