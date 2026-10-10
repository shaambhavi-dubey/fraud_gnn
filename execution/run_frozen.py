"""Run the frozen pipeline from the authorized execution commit, with a complete attempt log.

Put this file in the repo at execution/run_frozen.py and run from the repo root, venv active:

    python execution\\run_frozen.py --check          # verify git state only, run nothing
    python execution\\run_frozen.py                  # run every stage of run_all.sh in order
    python execution\\run_frozen.py --from shifts    # continue from a stage (run_seeds resumes by itself)
    python execution\\run_frozen.py --only plots     # a single stage

What it does and does not do:
  * refuses to start unless the code (fraudshift/, run_all.sh, lock, protocol, amendments) is identical to the
    execution commit and no tracked file is modified;
  * runs `python -m fraudshift.<stage>` exactly as run_all.sh lists them, one stage at a time;
  * writes one log per attempt to logs/ (never overwritten) and one line per attempt to
    logs/run_manifest.jsonl (commit, stage, start, end, seconds, exit code);
  * stops at the first failing stage. It never retries, never deletes outputs, never edits any result;
  * changes nothing about models, seeds, thresholds or metrics.
"""
import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXEC_COMMIT = "b57f61a4f484f8cb760f19b4af9e32aa4acf2993"      # authorized execution commit (Amendment 1c)
CODE_PATHS = ["fraudshift", "run_all.sh", "requirements-lock.txt", "requirements.txt", "PROTOCOL.md", "amendments"]
LOGS = ROOT / "logs"


def git(*args, check=False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def verify_state(exec_commit):
    ok = True
    if git("cat-file", "-e", f"{exec_commit}^{{commit}}").returncode:
        print(f"FAIL: execution commit {exec_commit} not found in this clone (git fetch?)")
        return False
    head = git("rev-parse", "HEAD", check=True).stdout.strip()
    for label, args in (("code differs from the execution commit", ["diff", "--quiet", exec_commit, "HEAD", "--", *CODE_PATHS]),
                        ("unstaged edits to tracked files", ["diff", "--quiet"]),
                        ("staged but uncommitted changes", ["diff", "--cached", "--quiet"])):
        if git(*args).returncode:
            print(f"FAIL: {label}")
            ok = False
    print(f"HEAD {head}\nexecution commit {exec_commit}\ncode identical and tree clean: {ok}")
    return ok


def stages_from_run_all():
    text = (ROOT / "run_all.sh").read_text()
    return re.findall(r"python -m fraudshift\.(\w+)", text)


def run_stage(stage, exec_commit):
    LOGS.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = LOGS / f"{stamp}_{stage}.log"
    head = git("rev-parse", "HEAD").stdout.strip()
    start, t0 = now(), time.time()
    print(f"\n=== {stage}  (log: {log.name}) ===", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"stage={stage} start={start} head={head} exec_commit={exec_commit}\n"
                 f"python={sys.version.split()[0]} platform={platform.platform()}\n")
        proc = subprocess.Popen([sys.executable, "-m", f"fraudshift.{stage}"], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        code = proc.wait()
        fh.write(f"\n[exit code {code}]\n")
    entry = dict(stage=stage, start=start, end=now(), seconds=round(time.time() - t0, 1), exit_code=code,
                 head=head, exec_commit=exec_commit, log=str(log.relative_to(ROOT)))
    with open(LOGS / "run_manifest.jsonl", "a", encoding="utf-8") as mf:
        mf.write(json.dumps(entry) + "\n")
    print(f"--- {stage}: exit {code}, {entry['seconds']} s", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify git state only")
    ap.add_argument("--only", help="run a single stage")
    ap.add_argument("--from", dest="start_from", help="start at this stage")
    ap.add_argument("--exec-commit", default=EXEC_COMMIT)
    a = ap.parse_args()

    if not verify_state(a.exec_commit):
        raise SystemExit("Refusing to run: fix the state above (do not edit code; ask first).")
    stages = stages_from_run_all()
    print("stages:", " -> ".join(stages))
    if a.check:
        return
    if a.only:
        stages = [a.only]
    elif a.start_from:
        stages = stages[stages.index(a.start_from):]
    for st in stages:
        if run_stage(st, a.exec_commit):
            raise SystemExit(f"\nStage {st} FAILED. Its log and manifest line are kept. Do not delete outputs, "
                             "do not rerun silently: report the failure, then decide how to continue.")
    print("\nAll requested stages finished.")


if __name__ == "__main__":
    main()