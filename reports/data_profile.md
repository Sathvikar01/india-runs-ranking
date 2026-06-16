# Data Profile

_Generated from `data/raw/candidates.jsonl` — 100,000 of 100,000 candidates._

## Pool size and shape
- Total candidates: **100,000**
- Profiled in this run: **100,000**
- Career-history entries: 1–10 per candidate (schema-bound)
- Education entries: 0–N per candidate
- Skills entries: 0–N per candidate

## Years of experience
- min: 1.00
- median: 6.80
- mean: 7.17
- max: 16.90

## Top current titles
- Business Analyst — 5,833
- HR Manager — 5,830
- Mechanical Engineer — 5,791
- Accountant — 5,764
- Project Manager — 5,754
- Customer Support — 5,750
- Operations Manager — 5,744
- Content Writer — 5,727
- Sales Executive — 5,713
- Civil Engineer — 5,702
- Graphic Designer — 5,689
- Marketing Manager — 5,524
- Software Engineer — 3,450
- Full Stack Developer — 2,873
- Cloud Engineer — 2,836
- Java Developer — 2,809
- .NET Developer — 2,788
- DevOps Engineer — 2,787
- Mobile Developer — 2,757
- Frontend Engineer — 2,738

## Top countries
- India — 75,113
- USA — 9,978
- Australia — 2,579
- Canada — 2,506
- UK — 2,472
- Germany — 2,469
- Singapore — 2,453
- UAE — 2,430

## Top locations (city, region)
- Bhubaneswar, Odisha — 4,321
- Noida, Uttar Pradesh — 4,283
- Hyderabad, Telangana — 4,283
- Jaipur, Rajasthan — 4,268
- Bangalore, Karnataka — 4,238
- Kolkata, West Bengal — 4,230
- Indore, Madhya Pradesh — 4,198
- Pune, Maharashtra — 4,186
- Chennai, Tamil Nadu — 4,164
- Delhi, Delhi — 4,161
- Trivandrum, Kerala — 4,151
- Ahmedabad, Gujarat — 4,143
- Chandigarh, Chandigarh — 4,128
- Coimbatore, Tamil Nadu — 4,113
- Vizag, Andhra Pradesh — 4,093
- Kochi, Kerala — 4,073
- Mumbai, Maharashtra — 4,043
- Gurgaon, Haryana — 4,037
- Sydney — 2,579
- San Francisco — 2,536

## Top industries
- IT Services — 29,881
- Software — 22,417
- Manufacturing — 22,305
- Conglomerate — 7,571
- Paper Products — 7,467
- Fintech — 2,808
- Food Delivery — 2,514
- E-commerce — 1,529
- Consulting — 1,274
- EdTech — 610
- SaaS — 328
- AI/ML — 278
- AdTech — 172
- Transportation — 162
- Insurance Tech — 155
- Gaming — 149
- HealthTech — 147
- HealthTech AI — 68
- Conversational AI — 62
- AI Services — 42

## Honeypot / trap signals
Each signal is the mean of the rule's 0-1 sub-score over the profiled subset. Candidates with any sub-score ≥ 0.5 are flagged: **99,971** of 100,000.

| Signal | Mean | Max |
|---|---:|---:|
| `skill_proficiency_vs_duration` | 0.000 | 1.000 |
| `yoe_vs_career_sum` | 0.015 | 0.955 |
| `perfect_skill_list_with_non_tech_title` | 0.000 | 0.000 |
| `multiple_current_positions` | 0.000 | 0.000 |
| `expert_in_too_many_skills` | 0.002 | 1.000 |
| `all_skills_zero_endorsements` | 0.000 | 0.000 |
| `high_skill_count_no_career_evidence` | 0.941 | 1.000 |

## Schema & integrity checks
- candidate_id format `CAND_XXXXXXX` is enforced at load time; no duplicates seen.
- career_history: min 1, max 10 entries; dates use ISO `YYYY-MM-DD`.
- skills: free-form strings, canonicalized via `src.preprocessing.normalize.normalize_skill`.
- redrob_signals: 23 fields, all present in profiled subset.
