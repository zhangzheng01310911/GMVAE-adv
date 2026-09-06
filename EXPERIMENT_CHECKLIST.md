# Experiment completion checklist

## Required inputs from authors

- Hong Kong processed or raw data and exact collection dates.
- PeMS-BAY and METR-LA arrays/adjacency used in the submission.
- Baseline source repositories and versions.
- Existing checkpoints/logs, if any.
- Exact graph-construction and missing-value rules.

## Minimum experiment matrix

1. Full revised model, seeds 0–9, all three datasets.
2. Strongest baseline under identical split, seeds 0–9.
3. Gaussian prior + KL.
4. GMM prior + KL.
5. Gaussian prior + Rényi.
6. GMM prior + Rényi.
7. K sensitivity: 2, 3, 4, 10, 20.
8. Alpha sensitivity: 0.75, 1.5, 1.75, 2, 5 (monitor estimator variance).
9. Verify the Gaussian Rényi finiteness condition for every reported run.
10. Congestion-threshold sensitivity and positive-class prevalence.

## Acceptance gates

- No manually typed table values.
- No excluded runs.
- Every mean/SD traceable to ten files.
- Same temporal split across compared methods.
- Predictions inverse-transformed before MAE/RMSE/MAPE.
- Table 2 highlighting generated programmatically.
- Table 4 and Table 5 matching when configurations match.
- All response claims contain a manuscript location.
