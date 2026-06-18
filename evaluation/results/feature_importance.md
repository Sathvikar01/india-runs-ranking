# Feature Importance

## Top 20 features by LightGBM gain (LTR model)

| rank | feature | gain | split count | gain % |
|---:|---|---:|---:|---:|
| 1 | `ai_keyword_hits_career` | 6379.4 | 1337 | 83.0 |
| 2 | `ai_career_share` | 510.6 | 118 | 6.6 |
| 3 | `yoe_in_5_9_band` | 128.3 | 538 | 1.7 |
| 4 | `product_company_count` | 104.3 | 834 | 1.4 |
| 5 | `has_ai_career_evidence` | 94.0 | 15 | 1.2 |
| 6 | `consulting_share` | 62.4 | 361 | 0.8 |
| 7 | `location_tier1_india` | 60.1 | 884 | 0.8 |
| 8 | `profile_views_30d` | 35.6 | 909 | 0.5 |
| 9 | `yoe_reported` | 35.5 | 700 | 0.5 |
| 10 | `willing_to_relocate` | 25.6 | 457 | 0.3 |
| 11 | `notice_period_score` | 25.0 | 349 | 0.3 |
| 12 | `yoe_career_sum` | 19.1 | 529 | 0.2 |
| 13 | `search_appearance_30d` | 18.1 | 673 | 0.2 |
| 14 | `location_is_noida_or_pune` | 17.4 | 425 | 0.2 |
| 15 | `profile_completeness` | 11.7 | 712 | 0.1 |
| 16 | `interview_completion_rate` | 11.4 | 497 | 0.1 |
| 17 | `github_activity_score` | 11.2 | 639 | 0.1 |
| 18 | `notice_period_days` | 10.6 | 142 | 0.1 |
| 19 | `recruiter_response_rate` | 10.0 | 478 | 0.1 |
| 20 | `avg_tenure_months` | 10.0 | 603 | 0.1 |

## Top 10 vs bottom 10 — feature mean difference

Positive delta → feature value is higher in the top-10 than the bottom-10.

| feature | top-10 mean | bottom-10 mean | delta |
|---|---:|---:|---:|
| `search_appearance_30d` | 391.9 | 157.0 | +234.900 |
| `profile_views_30d` | 91.4 | 61.3 | +30.100 |
| `github_activity_score` | 30.09 | 13.17 | +16.920 |
| `saved_by_recruiters_30d` | 23.8 | 9.4 | +14.400 |
| `n_skills` | 13.3 | 9.6 | +3.700 |
| `ai_keyword_hits_skills` | 5.4 | 2.4 | +3.000 |
| `advanced_skill_count` | 4.0 | 1.8 | +2.200 |
| `expert_skill_count` | 1.6 | 0.0 | +1.600 |
| `profile_completeness` | 68.24 | 66.84 | +1.400 |
| `n_named_jd_skills` | 2.6 | 1.3 | +1.300 |
| `ai_skill_count` | 2.9 | 1.6 | +1.300 |
| `ai_keyword_hits_career` | 3.4 | 2.3 | +1.100 |
| `n_ai_skill_advanced` | 1.4 | 0.5 | +0.900 |
| `has_offer_history` | 0.8 | 0.2 | +0.600 |
| `n_distinct_industries` | 2.3 | 1.7 | +0.600 |
| `endorsement_entropy` | 3.205 | 2.7425 | +0.463 |
| `offer_acceptance_rate` | 0.453 | 0.079 | +0.374 |
| `has_shipped_ranking_search_recsys` | 0.8 | 0.5 | +0.300 |
| `has_retrieval_ranking_evidence` | 0.8 | 0.5 | +0.300 |
| `n_education` | 1.6 | 1.3 | +0.300 |
