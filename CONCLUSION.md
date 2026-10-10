# Conclusion

Frozen comparison (protocol-v1, amendments 1, 1b, 1c). Results produced by the code at execution commit
`b57f61a4f484f8cb760f19b4af9e32aa4acf2993`. Every number below is in `results/raw/`; nothing was tuned.

## Main result: the GNN advantage is not supported, and the evidence points the other way

Temporal test (train on time steps 1-24, score 35-49; 16,670 labeled nodes, 1,083 illicit), primary feature set
(165 features), mean AUC-PR over 10 seeds:

| model | AUC-PR (sd) | recall | precision | FP rate among licit |
|---|---|---|---|---|
| XGBoost | 0.781 (0.003) | 0.713 | 0.825 | 0.011 |
| GraphSAGE | 0.407 (0.032) | 0.498 | 0.427 | 0.047 |
| GCN | 0.288 (0.057) | 0.393 | 0.334 | 0.064 |

Primary contrast, GraphSAGE minus XGBoost: -0.375 AUC-PR (seed-paired t-test, Holm p = 1.2e-10; Wilcoxon p = 0.002,
the smallest possible with 10 seeds; block bootstrap over 15 time steps, 95% interval [-0.507, -0.253]). Both tests exclude
zero, in the direction of XGBoost. The pre-specified criterion for a supported GNN advantage is therefore not met.
(`stats.py` prints "advantage supported by both tests" whenever both tests exclude zero; it does not encode direction.
Read the sign of the difference.) The 93-feature set gives the same ordering (0.746 / 0.370 / 0.213). Secondary:
GCN minus XGBoost -0.493 (bootstrap interval [-0.606, -0.357]); GraphSAGE minus GCN +0.118 ([0.028, 0.183]).
These results describe these implementations with the frozen configurations, not GNNs in general.

## Where it holds and where it does not (descriptive; no tests were pre-specified for these)

- **Drift dominates.** Window 35-42: AUC-PR XGBoost 0.908, GraphSAGE 0.533, GCN 0.404 (base rate 9.2%). Window 43-49,
  after the dark-market shutdown: 0.041 / 0.034 / 0.044 against a base rate of 2.5%. All three models are close to the
  base rate there, so the headline 0.781 is carried by the earlier window.
- **Connectivity groups** (in-degree; AUC-PR XGBoost / GraphSAGE / GCN): in0 0.427 / 0.277 / 0.188 (306 illicit),
  in1 0.951 / 0.494 / 0.397 (722), in2plus 0.139 / 0.128 / 0.056 (55). No group is underpowered (all above 30 illicit),
  but group base rates differ (5.3%, 8.5%, 2.3%), so AUC-PR is not comparable across groups. In in2plus all models are
  poor and the models are close.
- **Activity-mix shift** (in0 share raised from 35% to 75%): XGBoost 0.781 to 0.570, GraphSAGE 0.407 to 0.342, GCN 0.288
  to 0.244. XGBoost loses most in absolute terms and stays best at every level.
- **Prevalence shift:** recall and FP rate are constant by construction; AUC-PR and precision rise with prevalence
  (XGBoost 0.744 at 0.5x to 0.881 at 4x) and are not read as model change.

## Strict inductive evaluation (separate from the temporal test, never pooled)

4,721 held-out train-period nodes per seed (444 illicit, 9.4%). AUC-PR XGBoost 0.991 (sd 0.003), GraphSAGE 0.945 (0.008),
GCN 0.665 (0.169). Paired: GraphSAGE minus XGBoost -0.046 (Holm p 1.9e-8); GCN minus XGBoost -0.327 (3.3e-4); GraphSAGE
minus GCN +0.280 (4.7e-4). This is a random hold-out within one period, so there is no temporal shift; it must not be
compared with the temporal numbers as if it were the same population.

## Limits on what can be claimed

1. **Held-out isolation is only partly established.** Removing held-out rows and all their edges and fitting the scaler on
   retained rows only (checked by the preflight) establishes that part. It does not establish how the 72 precomputed
   one-hop neighbour-aggregate features were computed. The 165-feature inductive result is therefore not claimed to
   guarantee complete held-out-information isolation.
2. **Earlier exploration of Elliptic is disclosed.** [FILL IN: what earlier Elliptic work was done, when, and what it
   touched.] Before the freeze, a data audit and an in-degree audit were run using no model outcomes. New seeds do not
   make the dataset untouched.
3. **Seeds capture training randomness only.** The 10 paired seeds are not independent samples of the data, so seed-level
   p-values overstate precision; the block bootstrap over 15 time steps is the more conservative interval, and the time
   blocks are heterogeneous (see the post-shutdown window).
4. **GCN is unstable.** Validation AUC-PR splits into two clusters (about 0.51-0.60 and 0.82-0.83 for the primary feature
   set) and inductive AUC-PR into about 0.51-0.62 versus 0.90-0.91. In the logs the low-validation runs were mostly
   short (about 40-80 s, against 350-420 s for the high ones). The cause was not investigated and nothing was tuned; per-seed values are in the CSVs.
5. **Why the GNNs do not help here is untested.** One possibility is that the 72 aggregate features already give XGBoost
   one-hop neighbourhood information, but this study does not test it.

## Amendments, deviations and numerical changes

- `protocol-v1` and `PROTOCOL.md` are unchanged. `amendments/AMENDMENT_1.md` records all post-freeze corrections:
  (1) the `time` shadowing bug in run_seeds.py; (2) full 64-character SHA-256 values verified by check_data.py;
  (3) the strict inductive path rebuilt to fit the scaler once on retained nodes; (4) the in_degree() docstring and the
  audit-overwrite fix; (1b) float64 accumulation in `fit_scaler`, found by preflight v2 (protocol-v1's float32 scaler
  differed from a float64 reference by up to 2.04 standardized units).
  **Temporal-path features are therefore no longer bit-identical to the original implementation.**
- (1c) My first float64-reference check used an absolute 1e-4 bound, which failed at 1.7e-4 (float32 rounding at large
  values); it was replaced by a relative bound and disclosed in the amendment.
- Pre-freeze design changes (grouping by fixed in-degree bins, merged test windows 35-42 and 43-49) are in the protocol's
  deviations log.

## Run record

- Execution commit `b57f61a4f484f8cb760f19b4af9e32aa4acf2993`; run on HEAD df2ca39 (identical code). CPU only (8 cores),
  Python 3.14.5, torch 2.14.0+cpu, xgboost 3.4.1; torch printed a `torch.jit.script` FutureWarning on Python 3.14; no
  failures. No paid compute.
- Five wrapper invocations were made. Attempts 1-4 were stopped before finishing (no error in their logs); the fifth
  completed all eight stages with exit code 0. `run_seeds` resumes from existing outputs, so 27 of the 60 score files come
  from attempts 2-4. See `logs/ATTEMPTS.md`, `logs/run_manifest.jsonl` and the per-attempt logs. Interrupted runs were
  retrained from scratch with the same seed; no result was selected, discarded or rerun because of its value.
- `results/raw/compute_receipt.json`: 7,908 s of training across all 60 runs (it does not time shifts, inductive, stats
  or plots). Stage wall times of the complete attempt: 8,246 s. Hours worked: see `hours_log.md`.
- Not in git: `results/scores/` and `results/inductive/` (large, gitignored); they are regenerated by `run_all.sh`.