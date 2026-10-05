# AMENDMENT 1: pre-run, pre-outcome corrections to protocol-v1

Status: append-only. `PROTOCOL.md` and the tag `protocol-v1` (633487f35cc8946ed945acb599bb0f186ba2db16) are
NOT modified. This file records post-freeze corrections requested by the reviewer (Ryan Gomez) after
reviewing the preflight commit f7d016b2fb7e5ec12b9d8b6e5db3af3683ec3f44.

Scope statement: these are corrections to code defects, an unverifiable hash record, and a preprocessing-order
contract violation. They are not design changes. No model, threshold, shift, subgroup rule, seed, metric or
success criterion is changed. `fraudshift/config.py` is byte-identical to protocol-v1.

Outcome status: no model has been fit. `results/scores/`, `results/inductive/` and every outcome CSV listed in
PROTOCOL.md section 14 (other than `data_audit.csv`, which is a data audit) do not exist and were not opened.

## Correction 1: `run_seeds.py` time-module shadowing (code defect)
Line 23 of `fraudshift/run_seeds.py` at protocol-v1 was
`time= gte.time[lab], indeg=indeg[lab]`. It rebinds the name `time`, which shadows the imported `time`
module, so the first `time.time()` call would fail. `np.savez` already stores `time` and `indeg` explicitly,
so the line served no purpose. Fix: the line is deleted. No other line in the file changed.

## Correction 2: full SHA-256 values, verified by the data check
Section 3 of protocol-v1 lists 46-hex-character values, which are prefixes of the real 64-character digests.
Appended here, unchanged from the values captured by preflight-v1 (`preflight/preflight_receipt.json`,
commit f7d016b) and stored machine-readably in `amendments/pinned_data_hashes.json`:

| file | SHA-256 |
|---|---|
| elliptic_txs_features.csv | fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0 |
| elliptic_txs_classes.csv  | 93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493 |
| elliptic_txs_edgelist.csv | a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e |

Each protocol-v1 prefix is a prefix of the corresponding full value (checked by the preflight).
`fraudshift/check_data.py` now computes the three digests first and stops with a non-zero exit (fail closed)
on any missing file or mismatch, before reading any data.

## Correction 3: strict inductive preprocessing order (section 12 contract)
Protocol section 12 says held-out features are never used in preprocessing. At protocol-v1,
`build_splits()` standardized with the mean/std of ALL train-period nodes (including nodes later held out),
and `inductive.py` then re-standardized on retained nodes. The final features were algebraically the same,
but the literal contract was not met. Fix:
- `data.py` gains `build_raw_splits()` (unstandardized), `fit_scaler()` and `apply_scaler()`;
  `build_splits()` is rebuilt from them. Labels, edges, time steps and node ids are bit-identical to
  protocol-v1; the standardized features differ only by the scaler-accuracy fix of correction 5.
- New torch-free `fraudshift/inductive_split.py`: start from unstandardized train-period features, draw the
  hold-out, remove held-out nodes and all their incident edges, fit the scaler ONCE on retained nodes only,
  then apply it to the training graph, the validation graph and the inference graph.
- `inductive.py` uses this module. Hold-out draw (`default_rng(1000 + seed)`), stratification, model list,
  thresholding and metrics are unchanged.
- Check: for every seed 0-9 the preflight perturbs the raw features of the held-out nodes (a copy of the
  data) and verifies that the scaler and the training-graph features are bit-for-bit unchanged.

## Correction 5: scaler accumulated in float64 (found by preflight v2 on the real data)
Preflight v2 compared the corrected inductive features with the protocol-v1 two-step route and found a
maximum difference of 0.215 (tolerance 1e-3, a check added by the author, not by the protocol). Diagnosis
(read-only probes on the real features, `preflight/` diagnostics reported to the reviewer): the cause is float32
accumulation inside `fit_scaler`. On the 108,950 train-period nodes, `x.std(0)` in float32 has a maximum relative
error of 7.98e-4 against a float64 reference and `x.mean(0)` a maximum absolute error of 6.36e-4; the same
computation with a float64 accumulator is exact and column-by-column float32 reduction agrees to 1e-7.
Some columns have |value| up to about 260, so the standardized values differ by up to about 0.2. Protocol-v1's
`build_splits()` has the same defect (same arithmetic), so this predates the corrections and affects the temporal
path as well. It is not leakage: the independence checks pass bitwise.
Fix: `fit_scaler()` computes mean and standard deviation with `dtype=np.float64`; `apply_scaler()` is unchanged.
The same function is used by the temporal path and the inductive path, so the two stay consistent. Consequence:
temporal-path features are no longer bit-identical to protocol-v1; they are closer to the float64 reference.
Labels, edges, time steps, node ids and every model, threshold and metric are unchanged. No outcome exists.
Measured on the real data by preflight v2: over the train, validation and test features of the temporal path,
protocol-v1's float32 scaler differs from the float64 reference by up to 2.04 standardized units (heavy-tailed
columns reach |value| in the thousands); the corrected scaler differs by 1.7e-4, which is float32 output rounding at
that magnitude (about 6e-8 relative). Verification (preflight v2): standardized features of the temporal path and the
inductive training features equal a float64 reference within float32 rounding (rtol 1e-6, atol 1e-6), a bound about
1000 times tighter than the protocol-v1 scaler error; the largest differences and the distance to protocol-v1 are
recorded in the receipt. The first version of this check used an absolute bound of 1e-4, which float32 storage cannot
meet at |value| in the thousands; it was replaced by the relative bound after preflight v2 showed this, and the
change is recorded here and in its own commit.
If the reviewer prefers to keep the frozen float32 scaler, this correction can be reverted on its own commit.

## Correction 4: stale docstring and audit overwrite
- `Graph.in_degree()` docstring said "Total degree (in + out)"; the code computes in-degree only (as section 9
  requires). Docstring corrected; code unchanged.
- `check_data.py` no longer writes `results/raw/data_audit.csv`. It recomputes the audit in memory and compares
  with the committed file; on mismatch it writes `results/raw/data_audit_regenerated.csv` and exits non-zero,
  leaving the tracked file untouched.

## Exact diff
`amendments/AMENDMENT_1_code.diff` is `git diff protocol-v1 -- fraudshift` taken at the corrected commit.
Files touched in `fraudshift/`: check_data.py, data.py (also correction 5), inductive.py, run_seeds.py, and the new inductive_split.py.
Frozen-behavior files byte-identical to protocol-v1: config.py, models.py, metrics.py, subgroups.py, shifts.py,
stats.py, plots.py, compute_receipt.py, degree_audit.py, io.py (checked by the preflight).

## Verification
`preflight/run_preflight_v2.py` (outputs `preflight/preflight_v2_receipt.json` and `preflight_v2_log.txt`) checks
all of the above and the unchanged items from preflight-v1. It never fits a model and never opens outcome files.
