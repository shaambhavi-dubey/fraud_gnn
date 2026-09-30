# Fraud-GNN population-shift study (Elliptic)

Compares XGBoost, GCN and GraphSAGE on the Elliptic Bitcoin transaction dataset under a
temporal split, subgroup segmentation, deliberate population shifts and a strict inductive
(new-node) evaluation. Part of the FinanceMeta research sprint.

## Status
- Protocol frozen: tag `protocol-v1`, commit `633487f35cc8946ed945acb599bb0f186ba2db16`.
- Results: not yet produced. (Update this section at handoff with the result commit SHA.)
- Later commits are documentation or tooling only unless listed in the Deviations log of
  `PROTOCOL.md`. The design in `PROTOCOL.md` and `fraudshift/config.py` is not changed
  after the tag.

## Read this first
`PROTOCOL.md` defines everything: dataset pinning, split, preprocessing, seeds, threshold
rule, grouping rule, both evaluations, statistics and the execution boundary.

Key definitions:
- **Unit:** a Bitcoin transaction (a node is one transaction, not an account). There is no
  account-age field, and none is used or invented.
- **Grouping:** transaction connectivity = in-degree, the number of parent transactions in
  the same time step. Fixed bins in0 / in1 / in2plus, chosen from train-period counts.
  It is not account activity or age.
- **Temporal evaluation:** train on time steps 1-24, validate on 25-34, test on 35-49.
- **Strict inductive evaluation:** within the training period, hold out 20% of labeled
  nodes; their nodes and edges stay out of training and message passing. Reported
  separately from the temporal evaluation and never pooled with it.

## Layout
| Path | Contents |
|---|---|
| `PROTOCOL.md` | Frozen study design and Deviations log |
| `fraudshift/config.py` | Every protocol number (seeds, split, bins, shift levels) |
| `fraudshift/data.py` | Loading, temporal split, in-degree groups |
| `fraudshift/models.py`, `metrics.py` | XGBoost / GCN / GraphSAGE training, metrics |
| `fraudshift/check_data.py`, `degree_audit.py` | Data audits (no model outcomes) |
| `fraudshift/run_seeds.py` | Ten-seed runs on the temporal split |
| `fraudshift/subgroups.py`, `shifts.py`, `inductive.py` | Subgroups, shifts, inductive test |
| `fraudshift/stats.py`, `plots.py` | Paired tests, block bootstrap, figures |
| `fraudshift/compute_receipt.py` | Hardware, library versions, training time |
| `hours_log.md` | Working hours against the 14-hour ceiling |
| `requirements.txt`, `requirements-lock.txt` | Dependencies and the exact installed versions |

## Setup
Tested with Python 3.14 on Windows, CPU-only PyTorch. Exact versions are in
`requirements-lock.txt`.
```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Data is not included. Download the Elliptic data set (Kaggle: ellipticco/elliptic-data-set)
and place these three files in `data/elliptic/`, or set `ELLIPTIC_DIR` to their folder:
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.
Their SHA-256 hashes are in section 3 of `PROTOCOL.md`; check yours match.

## Reproduce
One command (Linux/macOS/Git Bash): `bash run_all.sh`

Equivalent on Windows PowerShell, run in this order:
```
python -m fraudshift.check_data
python -m fraudshift.run_seeds
python -m fraudshift.subgroups
python -m fraudshift.shifts
python -m fraudshift.inductive
python -m fraudshift.stats
python -m fraudshift.plots
python -m fraudshift.compute_receipt
```
`run_seeds` skips runs that already exist in `results/scores/`, so an interrupted run can
be resumed. `check_data` must report 165 feature columns, 49 time steps and 0 edges
crossing time steps, otherwise it stops.

## Outputs
| File | Contents |
|---|---|
| `results/raw/data_audit.csv` | Nodes and labels per time step |
| `results/raw/per_seed_metrics.csv` | One row per model x feature set x seed, full test set |
| `results/raw/per_group_metrics.csv` | Same, per in-degree group, with underpowered flag |
| `results/raw/shift_metrics.csv` | Temporal windows, activity-mix and prevalence shifts |
| `results/raw/inductive_metrics.csv` | Held-out-node results per model x seed |
| `results/raw/paired_tests.csv` | Paired t / Wilcoxon per contrast, Holm-adjusted |
| `results/raw/bootstrap_intervals.csv` | Block-bootstrap intervals over test time steps |
| `results/raw/inductive_paired_tests.csv` | Paired tests on the inductive result |
| `results/raw/compute_receipt.json` | Hardware, versions, training seconds |
| `results/plots/*.png` | Subgroup plots, shift curves, inductive bars |

Per-node scores go to `results/scores/` and `results/inductive/` (not committed).

## Limitations
- Nodes are transactions; there is no account-age field.
- Few labeled illicit nodes in the late time steps (for example 2 at step 46); groups or
  windows with fewer than 30 illicit test nodes are flagged underpowered.
- GCN and GraphSAGE message passing uses the whole time-step graph with edges in both
  directions, including later transactions. Only the grouping rule is creation-time safe.
- Illicit share of labeled nodes differs across splits (about 9.4% train, 19.8%
  validation, 6.5% test); the threshold is fixed on validation and not re-tuned.

## Earlier exploratory work
Earlier single-seed runs (synthetic UPI-style graphs and Elliptic) live in a separate
repository (`gnn-upi`). They are exploratory and pre-protocol, are not rerun, and are not
evidence in this study.