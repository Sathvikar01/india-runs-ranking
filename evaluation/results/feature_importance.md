# Feature Importance

## Top 20 features by LightGBM gain (LTR model)

| rank | feature | gain | split count | gain % |
|---:|---|---:|---:|---:|
| 1 | `yoe_in_5_9_band` | 1326.0 | 161 | 39.7 |
| 2 | `ai_keyword_hits_career` | 503.5 | 581 | 15.1 |
| 3 | `jd_keyword_count_career` | 321.6 | 324 | 9.6 |
| 4 | `seniority_distance_from_ideal` | 230.1 | 202 | 6.9 |
| 5 | `location_jd_match` | 96.5 | 195 | 2.9 |
| 6 | `location_tier1_india` | 94.9 | 507 | 2.8 |
| 7 | `availability_composite` | 85.8 | 660 | 2.6 |
| 8 | `has_distributed_systems_evidence` | 73.6 | 337 | 2.2 |
| 9 | `ai_career_share` | 73.4 | 76 | 2.2 |
| 10 | `behavioral_availability` | 50.2 | 682 | 1.5 |
| 11 | `seniority_jd_match` | 47.6 | 16 | 1.4 |
| 12 | `behavioral_negative` | 35.2 | 418 | 1.1 |
| 13 | `github_activity_score` | 29.8 | 408 | 0.9 |
| 14 | `open_to_work` | 28.4 | 322 | 0.8 |
| 15 | `product_company_count` | 25.9 | 357 | 0.8 |
| 16 | `location_is_noida_or_pune` | 25.3 | 134 | 0.8 |
| 17 | `has_ai_career_evidence` | 21.8 | 7 | 0.7 |
| 18 | `behavioral_positive` | 18.5 | 412 | 0.6 |
| 19 | `distributed_systems_count` | 18.4 | 131 | 0.6 |
| 20 | `search_appearance_30d` | 18.0 | 458 | 0.5 |

## Top 10 vs bottom 10 — feature mean difference

Positive delta → feature value is higher in the top-10 than the bottom-10.

| feature | top-10 mean | bottom-10 mean | delta |
|---|---:|---:|---:|
| `search_appearance_30d` | 407.1 | 237.5 | +169.600 |
| `profile_views_30d` | 97.6 | 54.4 | +43.200 |
| `profile_completeness` | 72.26 | 62.06 | +10.200 |
| `skill_endorsement_max` | 45.3 | 36.8 | +8.500 |
| `skill_endorsement_mean` | 17.0363 | 11.8983 | +5.138 |
| `current_role_tenure_months` | 30.9 | 26.3 | +4.600 |
| `saved_by_recruiters_30d` | 19.9 | 15.5 | +4.400 |
| `ai_keyword_hits_skills` | 4.4 | 1.4 | +3.000 |
| `ai_keyword_hits_career` | 3.3 | 0.6 | +2.700 |
| `jd_keyword_count_career` | 2.6 | 0.7 | +1.900 |
| `advanced_skill_count` | 3.9 | 2.0 | +1.900 |
| `n_named_jd_skills` | 2.4 | 0.5 | +1.900 |
| `jd_skill_match_count` | 2.4 | 0.5 | +1.900 |
| `n_skills` | 12.6 | 10.8 | +1.800 |
| `expert_skill_count` | 1.6 | 0.0 | +1.600 |
| `ai_skill_count` | 2.4 | 0.8 | +1.600 |
| `engagement_intensity` | 12.6509 | 11.6837 | +0.967 |
| `n_senior_roles` | 1.1 | 0.2 | +0.900 |
| `n_ai_skill_advanced` | 1.4 | 0.5 | +0.900 |
| `jd_skill_match_expert_count` | 0.8 | 0.0 | +0.800 |
