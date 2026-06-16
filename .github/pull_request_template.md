---
name: Pull Request
about: Submit a code change
title: "[PR] "
labels: []
assignees: []
---

## Summary

<!-- One-paragraph description of the change and why. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds capability)
- [ ] Breaking change (fix or feature that would change existing behavior)
- [ ] Documentation
- [ ] Refactor (no behavior change)
- [ ] Test

## How was it tested?

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Pipeline smoke run on a 100-row sample
- [ ] Ablation result on a 5 k dev split (attach metric)

## Checklist

- [ ] `ruff check src tests` passes
- [ ] `black --check src tests` passes
- [ ] `mypy src` passes
- [ ] `pytest -q` passes (coverage ≥ 90 %)
- [ ] No new dependency without justification
- [ ] No new network call inside `src/serving/`
- [ ] Docs updated where relevant
