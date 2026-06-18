# Reasoning Quality Audit

_Source: `outputs\dry_run\team_xxx_dryrun_20260618_003043.csv`_

## Summary

- Rows: **100**
- Clean rows (no issues): **63** (63.0 %)
- Rows with specific facts: **100** (100.0 %)
- Rows with JD connection: **63** (63.0 %)
- Rows with honest concerns: **100** (100.0 %)
- Rows with hallucination issues: **0** (0.0 %)
- Unique reasonings: **100** / 100
- Mean pairwise bigram Jaccard: **0.1636** (lower = more diverse)
- Length: min 113, max 248, mean 202.7
- Length violations: 0 below 50 chars, 0 above 320

### Stage 4 checks (per `submission_spec.md:75-95`)

| Check | Verdict |
|---|---|
| Specific facts (yoe/title/skills/signal values) | PASS (n=100/100) |
| JD connection (retrieval/ranking/LLM/etc.) | WARN (n=63/100) |
| Honest concerns where expected | PASS (n=100/100) |
| No hallucination | PASS (n=0/100) |
| Variation (no all-identical reasonings) | PASS (n=100/100) |
| Rank consistency (top positive, bottom has concerns) | PASS |

## Per-row issues

| Rank | candidate_id | issues | reasoning_len |
|---:|---|---:|---:|
| 1 | `CAND_0002025` | no_jd_connection | 231 |
| 3 | `CAND_0003841` | no_jd_connection | 113 |
| 5 | `CAND_0001707` | no_jd_connection | 198 |
| 6 | `CAND_0004479` | no_jd_connection | 188 |
| 9 | `CAND_0002283` | no_jd_connection | 215 |
| 11 | `CAND_0000789` | no_jd_connection | 199 |
| 22 | `CAND_0002120` | no_jd_connection | 179 |
| 23 | `CAND_0003506` | no_jd_connection | 207 |
| 25 | `CAND_0004674` | no_jd_connection | 203 |
| 29 | `CAND_0001089` | no_jd_connection | 215 |
| 31 | `CAND_0001870` | no_jd_connection | 183 |
| 37 | `CAND_0002611` | no_jd_connection | 196 |
| 44 | `CAND_0003950` | no_jd_connection | 193 |
| 45 | `CAND_0000981` | no_jd_connection | 210 |
| 47 | `CAND_0004245` | no_jd_connection | 163 |
| 50 | `CAND_0002247` | no_jd_connection | 204 |
| 51 | `CAND_0004283` | no_jd_connection | 204 |
| 52 | `CAND_0004398` | no_jd_connection | 181 |
| 53 | `CAND_0001787` | no_jd_connection | 196 |
| 55 | `CAND_0003568` | no_jd_connection | 184 |
| 56 | `CAND_0000486` | no_jd_connection | 193 |
| 58 | `CAND_0001588` | no_jd_connection | 194 |
| 59 | `CAND_0004976` | no_jd_connection | 218 |
| 62 | `CAND_0001492` | no_jd_connection | 211 |
| 63 | `CAND_0004903` | no_jd_connection | 182 |
| 67 | `CAND_0004223` | no_jd_connection | 197 |
| 69 | `CAND_0002761` | no_jd_connection | 204 |
| 70 | `CAND_0002266` | no_jd_connection | 190 |
| 75 | `CAND_0000860` | no_jd_connection | 203 |
| 78 | `CAND_0002415` | no_jd_connection | 194 |
| 80 | `CAND_0004582` | no_jd_connection | 201 |
| 81 | `CAND_0001235` | no_jd_connection | 201 |
| 83 | `CAND_0001056` | no_jd_connection | 212 |
| 90 | `CAND_0002146` | no_jd_connection | 182 |
| 96 | `CAND_0002526` | no_jd_connection | 202 |
| 97 | `CAND_0001506` | no_jd_connection | 203 |
| 99 | `CAND_0001026` | no_jd_connection | 202 |
