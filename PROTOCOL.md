# PROTOCOL: Fraud-GNN population-shift study (FinanceMeta sprint)

Status: to be frozen by the git tag `protocol-v1`. After that commit nothing below may
change. Any later change goes in the Deviations log with date and reason, and is
reported in the handoff.

Owner: Shaambhavi Dubey
Protocol commit: the commit tagged `protocol-v1` (its SHA is sent to the reviewer)
Result commit: recorded in the handoff email

## 1. Question and unit of analysis
Does the GNN advantage over a strong tabular baseline survive (a) subgroup
segmentation, (b) deliberate shifts in the test population, and (c) scoring of
unseen nodes? Where does it disappear?

Unit of analysis: a Bitcoin TRANSACTION (one node = one transaction). Nodes are not
accounts and there is no account age anywhere in this study.

## 2. Models (fixed)
XGBoost (features only), GCN (2 layers), GraphSAGE (2 layers, mean aggregator).
Earlier single-seed runs on synthetic UPI-style graphs and on Elliptic are
exploratory and pre-protocol. They are not rerun and not used as evidence here.

## 3. Dataset (pinned)
- Elliptic Bitcoin dataset (Kaggle: ellipticco/elliptic-data-set).
- Download date: 2026-10-01
- SHA-256: elliptic_txs_features.csv = FD7F83573443C9E302E371D3F110E3B6224160F5D1ED8A
  elliptic_txs_classes.csv  = 93E2E7B2405C735BA752BF6BA06B947561DEDDD1F5A8FC
  elliptic_txs_edgelist.csv = A35053BA68A98E4382CAE2BA65B9D9E36B23B6439E02DF
- Verified on load: 203,769 nodes; 234,355 edges; 165 model features (plus txId and a
  time-step column); 49 time steps; labels illicit 4,545 / licit 42,019 / unknown
  157,205; 0 edges cross time steps.
- Label file coding: '1' = illicit, '2' = licit, 'unknown'. Positive class = illicit.
- Unknown-label nodes stay in the graph for message passing but are excluded from
  training loss, threshold selection and every metric.
- Edge direction assumption (not verifiable from the files): the first column of the
  edge list is the source (parent) transaction and the second the destination (child).
- Known limitations: nodes are transactions, not accounts; no account-age field; very
  few labeled illicit nodes in late time steps (e.g. 2 at step 46).

## 4. Temporal split (fixed)
- Train: time steps 1-24. Validation: 25-34. Test: 35-49.
- Each split's graph uses only edges among its own time steps. Because edges never
  cross time steps, no edge connects a train node to a validation or test node.
- Labeled nodes (illicit / licit): train 2,219 / 21,387; validation 1,243 / 5,045;
  test 1,083 / 15,587.
- Illicit share of labeled nodes differs across splits: about 9.4% (train), 19.8%
  (validation), 6.5% (test). The threshold is chosen on validation and applied
  unchanged to test, so this difference is expected to raise false positives among
  licit cases on test. It is reported, not corrected.

## 5. Features and preprocessing
- Primary feature set: 165 features (93 local + 72 aggregated one-hop), time step
  excluded, same for all models. Ablation: 93 local features only.
- Standardization: mean/std fit on train-period nodes only (features only, no labels).
- GCN and GraphSAGE treat edges as undirected (symmetrized).
- Class imbalance: scale_pos_weight (XGBoost) / class-weighted loss (GNNs), computed
  from training labels only.

## 6. Training (fixed before outcomes)
- Seeds: 0-9.
- XGBoost: 500 trees max, depth 6, lr 0.1, subsample 0.8, colsample_bytree 0.8,
  early stopping on validation AUC-PR (30 rounds).
- GCN / GraphSAGE: hidden 128, 2 layers, dropout 0.5, lr 1e-3, weight decay 5e-4, max
  200 epochs, early stopping on validation AUC-PR (patience 20).
- No tuning on test data. Compute budget: at most 14 working hours in total for the
  sprint; runs on CPU unless a GPU is explicitly noted in the handoff.
- A compute receipt is part of the handoff: hardware, Python and library versions,
  per-run training seconds (from the code) and a working-hours log (manual).

## 7. Threshold (validation-only)
One threshold per model per seed, chosen to maximize F1 for the illicit class on the
validation set. Never re-chosen on test or under any shift.

## 8. Primary comparison
- Metric: AUC-PR (illicit) on the full test set.
- Primary contrast: GraphSAGE vs XGBoost, paired by seed.
- Secondary contrasts: GCN vs XGBoost, GraphSAGE vs GCN. Holm correction across the
  three. Also reported: F1, precision, recall at the validation threshold.

## 9. Grouping rule (subgroups)
- Quantity: transaction in-degree = number of parent transactions inside the same
  time step that feed this transaction. Named everywhere "transaction connectivity
  (in-degree)". It is NOT account activity and NOT account age.
- Groups (fixed bins, no data-dependent cutoffs): in0 (in-degree 0), in1 (in-degree
  1), in2plus (in-degree 2 or more).
- The bins are fixed from the training period and are never recomputed on validation or
  test data; every node is assigned to a bin by its own in-degree using those same bins.
- Why these bins: chosen from train-period labeled counts only (in0 7,405 nodes /
  582 illicit; in1 10,710 / 1,559; in2plus 5,491 / 78) before any model outcome.
  Finer bins were rejected because in-degrees 3 and above have too few illicit
  nodes. Terciles were rejected because most nodes have degree 2 or less.
- Temporal availability of the inputs: in-degree is known when a transaction is
  created (its inputs are fixed at that moment). Out-degree appears only when outputs
  are later spent, so it is not used for grouping.
- Limits: the count covers same-time-step parents only. GCN and GraphSAGE message
  passing uses the whole time-step graph with edges in both directions, including
  later transactions that spend from a node, so the models' own inputs are
  retrospective within a time step. The grouping rule is time-safe; the GNN inputs
  are not claimed to be creation-time-only.
- Per group per model per seed: AUC-PR, recall, FN rate, and false-positive rate
  among licit = FP / (FP + TN). A group or window with fewer than 30 illicit test
  nodes is flagged underpowered and is never merged or dropped after seeing results.

## 10. Two distinct evaluations (not interchangeable, never pooled)
A. Temporal evaluation (main test). Models train on time steps 1-24 and score the
   graphs of time steps 35-49. These nodes and edges are entirely new and come from
   a later period, so this measures generalization across time, including drift.
   Nodes are not seen in training, but graphs at test time come from a different
   period. Results are in per_seed_metrics, per_group_metrics and shift_metrics.
B. Strict inductive evaluation (section 12). Within the TRAINING period only, some
   labeled nodes are held out and scored as new nodes joining a known graph. There
   is no time difference. This measures whether a model can score unseen nodes that
   attach to a graph it was trained on.
Conclusions about "new nodes" must say which evaluation they come from.

## 11. Population shifts (evaluation A, test set; model and threshold fixed)
1. Natural temporal shift: test windows 35-42 and 43-49 (a large dark-market shutdown
   occurs around step 43). Windows were merged to 43-49 because step 43 alone has
   too few illicit nodes (24).
2. Activity-mix shift: resample test nodes so the share of in0 transactions moves
   through {original, 25%, 50%, 75%} within each class, overall illicit prevalence
   held fixed.
3. Prevalence shift: resample so the illicit share moves through {0.5x, 1x, 2x, 4x}
   of test prevalence (capped at 50%), within-class mix unchanged.
- Per level: FP rate among licit, FN rate, precision, recall, AUC-PR. FP rate among
  licit is reported separately from prevalence changes; precision shifts with
  prevalence even if the model is unchanged, so it is not read as model degradation.
- 200 resamples per level per seed.

## 12. Strict inductive evaluation (evaluation B)
- Per seed, hold out 20% of LABELED train-period nodes (stratified by label). The same
  hold-out is used for every model.
- Training graph: held-out nodes and ALL their incident edges removed. Held-out labels
  and features are never used in training, preprocessing decisions or message passing.
- Inference graph: held-out nodes are added back with edges ONLY to retained
  train-period nodes. Edges between two held-out nodes are dropped. No validation or
  test nodes are in this graph.
- XGBoost scores the same held-out nodes from features alone.
- Automatic checks in code: no held-out node id is in the training graph; edge indices
  are valid for the retained nodes.
- Report AUC-PR, recall, FP rate among licit on held-out nodes. Models are compared
  with each other on the same held-out nodes, paired by seed. Comparison with the
  main-test numbers is descriptive only (different population).

## 13. Statistics
- Seed variation: paired differences over 10 seeds (paired t-test and Wilcoxon),
  effect sizes. Seeds capture training randomness only.
- Independent data units: block bootstrap over test time steps (15 blocks) with node
  resampling inside blocks, 2000 resamples, 95% intervals on the paired difference.
- A GNN advantage is called supported only if the Holm-adjusted paired test AND the
  bootstrap interval both exclude zero for the primary contrast. Otherwise it is
  reported as inconclusive or null.

## 14. Outputs
results/raw: data_audit.csv, per_seed_metrics.csv, per_group_metrics.csv,
shift_metrics.csv, inductive_metrics.csv, paired_tests.csv, bootstrap_intervals.csv,
inductive_paired_tests.csv, compute_receipt.json; results/plots: subgroup plots, shift
curves, inductive bars; hours_log.md; README with one reproduction command;
CONCLUSION.md (short, keeps null or negative results).

## 15. Execution boundary
- Outcome runs start only after the protocol commit is tagged and sent for review.
- If review or compute delays completion, a runnable partial package plus the blocker is
  handed off. The frozen design is not altered to fit, and no step is bypassed.
- Any independent methods review required before an applicable held-out gate is
  completed before that gate. Whether a gate applies is confirmed with the reviewer.
- Older exploratory work (repo gnn-upi) is kept separate from this sprint's results.

## 16. Out of scope
Latency/Kafka only if the core handoff is finished. Synthetic UPI graphs are not used.

## Deviations log
Pre-freeze design changes (made before the protocol commit, after the data audit, which
used label counts and degree tables from the TRAIN period and per-time-step label counts,
but no model outcomes):
- Grouping changed from terciles of total degree to fixed in-degree bins. Reason:
  terciles collapsed (cutoffs 2 and 2) and total degree uses out-edges that are not
  known at transaction creation time.
- Temporal windows changed from {35-42, 43, 44-49} to {35-42, 43-49}. Reason: step 43
  alone had 24 illicit test nodes, below the 30-node floor. This decision was made after
  seeing per-time-step label counts.
Post-freeze deviations: (none yet)