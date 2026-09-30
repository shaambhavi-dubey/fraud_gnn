# Fraud-GNN population-shift study (FinanceMeta sprint)

Compares XGBoost, GCN and GraphSAGE on Elliptic under a temporal split, subgroup
segmentation, deliberate population shifts and an inductive new-node test.

Reproduce everything: `bash run_all.sh`  (needs the Elliptic files, see Step 2).

## Exact steps

### Step 1. Create the canonical repo
```bash
cd fraud-shift-study
git init
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git   # create the empty repo on GitHub first
```

### Step 2. Get the data and point the code at it
Local machine:
```bash
mkdir -p data/elliptic          # put the 3 csv files here
```
Kaggle notebook: attach the dataset "Elliptic Data Set", then find the folder:
```bash
find /kaggle/input -name "*edgelist*"
export ELLIPTIC_DIR=<the folder that find printed>
```
The three files must be named exactly:
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.

### Step 3. Install
```bash
pip install -r requirements.txt
```
(`torch_geometric` sometimes needs an install command matching your torch version. On
Kaggle try `pip install torch_geometric` first.)

### Step 4. Audit the data (allowed before freezing)
```bash
python -m fraudshift.check_data
```
Read every printed line. It must show 165 feature columns, 49 time steps, and 0 edges
crossing time steps, or the script stops. If anything differs from PROTOCOL.md, fix the
protocol NOW (before Step 6). Also note how many illicit nodes each time step has.

### Step 5. Fill in the protocol blanks
```bash
sha256sum data/elliptic/*.csv          # macOS: shasum -a 256 data/elliptic/*.csv
date +%F
```
Paste the date and the three hashes into section 3 of `PROTOCOL.md`. Read the whole
file and change anything you disagree with. After Step 6 it is frozen.

### Step 6. Freeze the protocol (commit BEFORE any model runs)
Do not run Step 7 yet.
```bash
git add PROTOCOL.md README.md requirements.txt run_all.sh .gitignore fraudshift results/raw/data_audit.csv
git commit -m "Freeze protocol v1 before any outcome runs"
git tag protocol-v1
git push -u origin main --tags
git rev-parse HEAD                     # <- this is the PROTOCOL SHA; write it in PROTOCOL.md header later
```

### Step 7. Run the core experiment
```bash
python -m fraudshift.run_seeds         # 3 models x 2 feature sets x 10 seeds
```
Progress prints one line per run. It skips finished runs if you restart it (safe on
Kaggle session timeouts). Output: `results/raw/per_seed_metrics.csv` and per-node scores
in `results/scores/` (not committed).

### Step 8. Subgroups, shifts, inductive test
```bash
python -m fraudshift.subgroups
python -m fraudshift.shifts
python -m fraudshift.inductive
```

### Step 9. Statistics and plots
```bash
python -m fraudshift.stats
python -m fraudshift.plots
```
`stats` prints the PRIMARY verdict line at the end. Copy it into CONCLUSION.md.

### Step 10. Write CONCLUSION.md (short)
Answer four things: (1) primary comparison result with the Holm p-value AND the
bootstrap interval; (2) which subgroups the advantage holds in, and which are
underpowered; (3) how FP rate among licit and FN rate moved under each shift; (4)
inductive result. State the Elliptic limitations (no account age, transactions not
accounts, few labeled illicit after time step 43).

### Step 11. Commit results and hand off
```bash
git add results/raw results/plots CONCLUSION.md PROTOCOL.md
git commit -m "Results for protocol v1"
git push
git rev-parse HEAD                     # <- RESULT SHA
git rev-parse protocol-v1              # <- PROTOCOL SHA
```
Email Ryan: repo URL, both SHAs, `bash run_all.sh`, the CSV list below, plots,
CONCLUSION.md. If something did not finish, send what runs plus the blocker.

## What each output file is
| File | Contents |
|---|---|
| results/raw/data_audit.csv | nodes and labels per time step |
| results/raw/per_seed_metrics.csv | one row per model x feature set x seed on the full test set |
| results/raw/per_group_metrics.csv | same, per degree-tercile group, underpowered flag |
| results/raw/shift_metrics.csv | temporal windows, activity-mix and prevalence shifts |
| results/raw/inductive_metrics.csv | held-out-node results per model x seed |
| results/raw/paired_tests.csv | paired t / Wilcoxon per contrast, Holm-adjusted |
| results/raw/bootstrap_intervals.csv | block-bootstrap CI over test time steps |
| results/raw/inductive_paired_tests.csv | paired tests on the inductive result |
| results/plots/*.png | subgroup plots, shift curves, inductive bars |

## Rules while running
- Never change config.py after Step 6. If you must, add a dated line to the
  Deviations log in PROTOCOL.md and say so in the handoff.
- Never choose a threshold, model or setting by looking at test results.
