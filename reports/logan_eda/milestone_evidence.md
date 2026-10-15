# Logan EDA evidence for the milestone

Source snapshot: 2026-10-12T02:04:44.317934+00:00.
Raw size: 532,376 rows and 372 columns.
Filtered size: 7,433 rows and 373 columns, including the added target.

Proposed chronological split (pending team agreement):
     split   seasons  examples  games
  training 2015–2022      4887   1942
validation      2023       823    272
      test 2024–2025      1723    547

All findings below concern training seasons 2015–2022 only.
- Training conversion rate: 51.83% across 4,887 attempts.
- Attempts requiring at most two yards: 52.10%.
- Conversion rate for ≤1 yard: 65.70%; 11+ yards: 22.37%.
- Missing entries across proposed training predictors: 0.
- Duplicate training game/play identifiers: 0.
- Domain-range flags (may overlap): 0.
- Largest absolute numeric target correlation: ydstogo, r=-0.271.

Use 01_feature_distributions, 02_conversion_relationships, and 03_season_patterns as candidate milestone figures; use 04_correlations if space permits. Refer to notebook captions for interpretation and caveats. No model-based importance or model results are claimed.

Limitations: selected conversion attempts; all penalties excluded; unadjusted associations; within-game dependence; proposed split requires team agreement. Conversion probability alone does not answer whether going for it maximizes win probability.
