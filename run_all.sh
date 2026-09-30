#!/usr/bin/env bash
# One command to reproduce everything:  bash run_all.sh
set -euo pipefail
python -m fraudshift.check_data
python -m fraudshift.run_seeds
python -m fraudshift.subgroups
python -m fraudshift.shifts
python -m fraudshift.inductive
python -m fraudshift.stats
python -m fraudshift.plots
