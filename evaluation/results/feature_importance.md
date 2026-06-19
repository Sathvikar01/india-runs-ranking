# Feature Importance

## Top 20 features by LightGBM gain (LTR model)

| rank | feature | gain | split count | gain % |
|---:|---|---:|---:|---:|
| 1 | `yoe_in_5_9_band` | 1317.7 | 157 | 39.5 |
| 2 | `ai_keyword_hits_career` | 561.3 | 538 | 16.8 |
| 3 | `seniority_distance_from_ideal` | 236.9 | 262 | 7.1 |
| 4 | `jd_keyword_count_career` | 233.5 | 195 | 7.0 |
| 5 | `location_tier1_india` | 101.0 | 574 | 3.0 |
| 6 | `location_jd_match` | 94.5 | 243 | 2.8 |
| 7 | `has_distributed_systems_evidence` | 85.7 | 406 | 2.6 |
| 8 | `availability_composite` | 79.5 | 514 | 2.4 |
| 9 | `behavioral_negative` | 55.0 | 343 | 1.6 |
| 10 | `ai_career_share` | 51.4 | 76 | 1.5 |
| 11 | `behavioral_availability` | 50.7 | 633 | 1.5 |
| 12 | `seniority_jd_match` | 46.2 | 15 | 1.4 |
| 13 | `location_is_noida_or_pune` | 35.8 | 142 | 1.1 |
| 14 | `github_activity_score` | 31.6 | 555 | 0.9 |
| 15 | `has_ai_career_evidence` | 24.8 | 9 | 0.7 |
| 16 | `behavioral_positive` | 21.3 | 479 | 0.6 |
| 17 | `open_to_work` | 18.7 | 332 | 0.6 |
| 18 | `skill_endorsement_mean` | 18.0 | 389 | 0.5 |
| 19 | `search_appearance_30d` | 18.0 | 418 | 0.5 |
| 20 | `distributed_systems_count` | 17.6 | 123 | 0.5 |

## Top 10 vs bottom 10 — feature mean difference

Positive delta → feature value is higher in the top-10 than the bottom-10.

| feature | top-10 mean | bottom-10 mean | delta |
|---|---:|---:|---:|
| `search_appearance_30d` | 407.1 | 194.4 | +212.700 |
| `profile_views_30d` | 97.6 | 62.7 | +34.900 |
| `skill_endorsement_max` | 45.3 | 25.8 | +19.500 |
| `profile_completeness` | 72.26 | 60.64 | +11.620 |
| `current_role_tenure_months` | 30.9 | 22.1 | +8.800 |
| `skill_endorsement_mean` | 17.0363 | 9.7715 | +7.265 |
| `saved_by_recruiters_30d` | 19.9 | 15.3 | +4.600 |
| `ai_keyword_hits_skills` | 4.4 | 1.1 | +3.300 |
| `advanced_skill_count` | 3.9 | 1.0 | +2.900 |
| `n_skills` | 12.6 | 10.1 | +2.500 |
| `ai_keyword_hits_career` | 3.3 | 0.9 | +2.400 |
| `jd_keyword_count_career` | 2.6 | 0.8 | +1.800 |
| `expert_skill_count` | 1.6 | 0.0 | +1.600 |
| `jd_skill_match_count` | 2.4 | 0.8 | +1.600 |
| `ai_skill_count` | 2.4 | 0.8 | +1.600 |
| `n_named_jd_skills` | 2.4 | 0.8 | +1.600 |
| `n_senior_roles` | 1.1 | 0.0 | +1.100 |
| `n_ai_skill_advanced` | 1.4 | 0.3 | +1.100 |
| `engagement_intensity` | 12.6509 | 11.7333 | +0.918 |
| `jd_skill_match_expert_count` | 0.8 | 0.0 | +0.800 |
