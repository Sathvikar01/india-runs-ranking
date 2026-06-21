# Reasoning Quality Audit

_Source: `outputs\dry_run\team_xxx_dryrun_20260621_024330.csv`_

## Summary

- Rows: **100**
- Clean rows (no issues): **57** (57.0 %)
- Rows with specific facts: **100** (100.0 %)
- Rows with JD connection: **57** (57.0 %)
- Rows with honest concerns: **100** (100.0 %)
- Rows with hallucination issues: **0** (0.0 %)
- Unique reasonings: **100** / 100
- Mean pairwise bigram Jaccard: **0.1515** (lower = more diverse)
- Length: min 123, max 246, mean 204.9
- Length violations: 0 below 50 chars, 0 above 320

### Stage 4 checks (per `submission_spec.md:75-95`)

| Check | Verdict |
|---|---|
| Specific facts (yoe/title/skills/signal values) | PASS (n=100/100) |
| JD connection (retrieval/ranking/LLM/etc.) | WARN (n=57/100) |
| Honest concerns where expected | PASS (n=100/100) |
| No hallucination | PASS (n=0/100) |
| Variation (no all-identical reasonings) | PASS (n=100/100) |
| Rank consistency (top positive, bottom has concerns) | PASS |

## Per-row issues

| Rank | candidate_id | issues | reasoning_len |
|---:|---|---:|---:|
| 1 | `CAND_0000031` | no_jd_connection | 230 |
| 3 | `CAND_0000319` | no_jd_connection | 173 |
| 8 | `CAND_0002025` | no_jd_connection | 232 |
| 10 | `CAND_0000422` | no_jd_connection | 225 |
| 11 | `CAND_0000666` | no_jd_connection | 201 |
| 12 | `CAND_0001056` | no_jd_connection | 194 |
| 22 | `CAND_0002438` | no_jd_connection | 175 |
| 23 | `CAND_0002876` | no_jd_connection | 216 |
| 25 | `CAND_0004824` | no_jd_connection | 206 |
| 26 | `CAND_0003382` | no_jd_connection | 169 |
| 30 | `CAND_0000705` | no_jd_connection | 218 |
| 31 | `CAND_0000981` | no_jd_connection | 210 |
| 34 | `CAND_0001930` | no_jd_connection | 214 |
| 35 | `CAND_0002211` | no_jd_connection | 212 |
| 37 | `CAND_0003977` | no_jd_connection | 199 |
| 39 | `CAND_0004520` | no_jd_connection | 183 |
| 43 | `CAND_0001218` | no_jd_connection | 209 |
| 44 | `CAND_0001302` | no_jd_connection | 188 |
| 45 | `CAND_0001494` | no_jd_connection | 221 |
| 47 | `CAND_0001651` | no_jd_connection | 184 |
| 48 | `CAND_0001707` | no_jd_connection | 179 |
| 51 | `CAND_0002120` | no_jd_connection | 210 |
| 52 | `CAND_0002344` | no_jd_connection | 192 |
| 53 | `CAND_0002770` | no_jd_connection | 206 |
| 55 | `CAND_0003145` | no_jd_connection | 190 |
| 56 | `CAND_0003290` | no_jd_connection | 210 |
| 57 | `CAND_0003506` | no_jd_connection | 188 |
| 62 | `CAND_0004112` | no_jd_connection | 210 |
| 70 | `CAND_0003841` | no_jd_connection | 123 |
| 72 | `CAND_0004555` | no_jd_connection | 214 |
| 75 | `CAND_0004243` | no_jd_connection | 198 |
| 78 | `CAND_0001940` | no_jd_connection | 202 |
| 80 | `CAND_0002184` | no_jd_connection | 199 |
| 81 | `CAND_0000647` | no_jd_connection | 204 |
| 83 | `CAND_0004031` | no_jd_connection | 194 |
| 86 | `CAND_0004275` | no_jd_connection | 205 |
| 88 | `CAND_0003022` | no_jd_connection | 188 |
| 92 | `CAND_0002524` | no_jd_connection | 198 |
| 94 | `CAND_0000588` | no_jd_connection | 204 |
| 95 | `CAND_0000599` | no_jd_connection | 200 |
| 96 | `CAND_0000718` | no_jd_connection | 190 |
| 97 | `CAND_0001323` | no_jd_connection | 204 |
| 99 | `CAND_0004754` | no_jd_connection | 192 |
