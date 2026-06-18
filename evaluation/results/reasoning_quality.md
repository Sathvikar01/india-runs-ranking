# Reasoning Quality Audit

_Source: `C:\Users\arsat\OneDrive\Desktop\india\india-runs-ranking\evaluation\results\submission.csv`_

## Summary

- Rows: **100**
- Clean rows (no issues): **53** (53.0 %)
- Rows with specific facts: **100** (100.0 %)
- Rows with JD connection: **53** (53.0 %)
- Rows with honest concerns: **99** (99.0 %)
- Rows with hallucination issues: **0** (0.0 %)
- Unique reasonings: **100** / 100
- Mean pairwise bigram Jaccard: **0.1447** (lower = more diverse)
- Length: min 147, max 237, mean 203.8
- Length violations: 0 below 50 chars, 0 above 320

### Stage 4 checks (per `submission_spec.md:75-95`)

| Check | Verdict |
|---|---|
| Specific facts (yoe/title/skills/signal values) | PASS (n=100/100) |
| JD connection (retrieval/ranking/LLM/etc.) | WARN (n=53/100) |
| Honest concerns where expected | PASS (n=99/100) |
| No hallucination | PASS (n=0/100) |
| Variation (no all-identical reasonings) | PASS (n=100/100) |
| Rank consistency (top positive, bottom has concerns) | PASS |

## Per-row issues

| Rank | candidate_id | issues | reasoning_len |
|---:|---|---:|---:|
| 1 | `CAND_0000031` | no_jd_connection | 230 |
| 5 | `CAND_0001930` | no_jd_connection | 215 |
| 6 | `CAND_0000319` | no_jd_connection | 173 |
| 8 | `CAND_0001505` | no_jd_connection | 215 |
| 9 | `CAND_0002025` | no_jd_connection | 231 |
| 10 | `CAND_0000273` | no_jd_connection | 194 |
| 11 | `CAND_0000422` | no_jd_connection | 227 |
| 12 | `CAND_0000666` | no_jd_connection | 170 |
| 22 | `CAND_0006551` | no_jd_connection | 173 |
| 25 | `CAND_0006076` | no_jd_connection | 217 |
| 26 | `CAND_0001458` | no_jd_connection | 178 |
| 29 | `CAND_0003382` | no_jd_connection | 201 |
| 30 | `CAND_0003724` | no_jd_connection | 195 |
| 31 | `CAND_0004398` | no_jd_connection | 202 |
| 35 | `CAND_0000705` | no_jd_connection | 217 |
| 37 | `CAND_0000981` | no_jd_connection | 189 |
| 39 | `CAND_0001870` | no_jd_connection | 162 |
| 43 | `CAND_0003977` | no_jd_connection | 221 |
| 44 | `CAND_0004131` | no_jd_connection | 190 |
| 48 | `CAND_0001218` | no_jd_connection | 183 |
| 49 | `CAND_0001302` | no_jd_connection | 188 |
| 50 | `CAND_0001494` | no_jd_connection | 219 |
| 51 | `CAND_0001651` | no_jd_connection | 205 |
| 52 | `CAND_0001940` | no_jd_connection | 195 |
| 53 | `CAND_0002770` | no_jd_connection | 206 |
| 55 | `CAND_0003290` | no_jd_connection | 209 |
| 57 | `CAND_0003506` | no_jd_connection | 188 |
| 59 | `CAND_0003756` | no_jd_connection | 222 |
| 62 | `CAND_0004031` | no_jd_connection | 207 |
| 63 | `CAND_0004112` | no_jd_connection | 188 |
| 67 | `CAND_0005421` | no_jd_connection | 183 |
| 69 | `CAND_0005509` | no_jd_connection | 147 |
| 70 | `CAND_0005538` | no_jd_connection | 186 |
| 71 | `CAND_0005685` | no_jd_connection | 221 |
| 72 | `CAND_0006354` | no_jd_connection | 215 |
| 75 | `CAND_0006538` | no_jd_connection | 195 |
| 76 | `CAND_0001131` | no_jd_connection | 207 |
| 79 | `CAND_0002415` | no_jd_connection | 200 |
| 80 | `CAND_0002793` | no_jd_connection | 195 |
| 81 | `CAND_0002852` | no_jd_connection | 211 |
| 88 | `CAND_0006445` | no_jd_connection | 195 |
| 93 | `CAND_0004383` | no_jd_connection | 202 |
| 94 | `CAND_0004480` | no_jd_connection | 198 |
| 95 | `CAND_0004674` | no_jd_connection | 190 |
| 96 | `CAND_0004913` | no_jd_connection | 197 |
| 98 | `CAND_0005191` | no_jd_connection | 195 |
| 100 | `CAND_0005919` | no_jd_connection | 192 |
