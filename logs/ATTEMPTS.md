# Attempt ledger for the frozen run

Execution commit: b57f61a4f484f8cb760f19b4af9e32aa4acf2993 (Amendment 1c).
All attempts used `python execution/run_frozen.py` on HEAD df2ca3947f6a822b89080621e2cd85f431d8033d, whose code is
identical to the execution commit (verified by the wrapper at the start of each attempt).
The wrapper writes a manifest line only after a stage ends, so attempts that were stopped have logs but no manifest lines.
Times are UTC (India time is UTC+5:30).

| # | Start (UTC) | Reached | Ended | Outputs created | Reason it stopped |
|---|---|---|---|---|---|
| 1 | 2026-10-05 17:29:49 | check_data | stopped before finishing; its log contains only the header (199 bytes) | none | 
| 2 | 2026-10-05 17:31:11 | run_seeds | interrupted after gcn all165 seed 7 | xgboost all165 seeds 0-9; gcn all165 seeds 0-7 (18 files) |  
| 3 | 2026-10-05 18:01:44 | run_seeds | interrupted after graphsage all165 seed 0 | gcn all165 seeds 8-9; graphsage all165 seed 0 (3 files) |  
| 4 | 2026-10-05 18:13:57 | run_seeds | interrupted after graphsage all165 seed 6 | graphsage all165 seeds 1-6 (6 files) |
| 5 | 2026-10-05 18:36:08 | all 8 stages | completed, every stage exit code 0 (manifest) | graphsage all165 seeds 7-9; all 30 local93 runs (33 files); all later stages | none |

`run_seeds` skips any run whose output file already exists, so attempts 3-5 resumed from the files left by earlier attempts.
A run that was in progress when an attempt was interrupted was lost and retrained from scratch with the same seed in a later
attempt. No result was selected, discarded or rerun because of its value.

Log evidence: the logs of attempts 2-4 end on ordinary result lines with no traceback and no "[exit code]" line (a finished
stage writes one), so those attempts were stopped from outside the process, not by a Python error. The log of attempt 1
contains only its header.

Accounting: `results/raw/compute_receipt.json` reports 7,908 s of training across all 60 runs, including the 27 made in
attempts 2-4. The wall-clock time of the complete attempt 5 was 8,246 s (manifest). Hands-on hours are in `hours_log.md`.