#!/usr/bin/env bash
# Reproduce the full study (needs the Elliptic files, see README).  Usage: bash run_all.sh
set -euo pipefail
python -m fraudshift.check_data
python -m fraudshift.run_seeds
python -m fraudshift.subgroups
python -m fraudshift.shifts
python -m fraudshift.inductive
python -m fraudshift.stats
python -m fraudshift.plots
python -m fraudshift.compute_receipt