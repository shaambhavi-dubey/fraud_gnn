"""Preflight v2: verification of AMENDMENT_1 (pre-run corrections). Run from the repo root:

    python preflight/run_preflight_v2.py

Run it AFTER committing the corrected code (the tree must be clean except preflight/ outputs).
Writes preflight/preflight_v2_receipt.json and preflight/preflight_v2_log.txt (v1 files are left untouched).

Boundary: never fits a model, never imports fraudshift.models / run_seeds / shifts / inductive / stats / plots,
never opens results/scores, results/inductive or any outcome CSV. Reports counts, hashes, versions and
pass/fail only. No metric, loss, prediction or class-conditional score is computed or written.
"""
import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraudshift import check_data  # noqa: E402
from fraudshift import config as C  # noqa: E402
from fraudshift.data import Graph, assign_group, build_raw_splits, build_splits, fit_scaler, load_full  # noqa: E402
from fraudshift.inductive_split import make_inductive_split  # noqa: E402

LOG, RECEIPT = [], {}
FROZEN_FILES = ["PROTOCOL.md", "fraudshift/config.py", "fraudshift/models.py", "fraudshift/metrics.py",
                "fraudshift/subgroups.py", "fraudshift/shifts.py", "fraudshift/stats.py", "fraudshift/plots.py",
                "fraudshift/compute_receipt.py", "fraudshift/degree_audit.py", "fraudshift/io.py",
                "fraudshift/__init__.py"]
ALLOWED_CHANGED = {"fraudshift/check_data.py", "fraudshift/data.py", "fraudshift/inductive.py",
                   "fraudshift/inductive_split.py", "fraudshift/run_seeds.py", "README.md", "run_all.sh"}
ALLOWED_PREFIXES = ("amendments/", "preflight/")
EXPECTED_ORDER = ["check_data", "run_seeds", "subgroups", "shifts", "inductive", "stats", "plots",
                  "compute_receipt"]
OUTCOME_CSVS = {"per_seed_metrics.csv", "per_group_metrics.csv", "shift_metrics.csv", "inductive_metrics.csv",
                "paired_tests.csv", "bootstrap_intervals.csv", "inductive_paired_tests.csv",
                "compute_receipt.json"}


def say(msg):
    print(msg)
    LOG.append(msg)


def check(name, ok, detail=""):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    RECEIPT.setdefault("checks", {})[name] = {"pass": bool(ok), "detail": str(detail)}
    return ok


def git(*args, strip=True):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout
    return out.strip() if strip else out


def exits_closed(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit as e:
        return isinstance(e.code, str) and e.code.startswith("FAIL")
    return False


# ---------------------------------------------------------------- 1. environment
def env_check():
    say("== 1. environment and dependency lock")
    RECEIPT["environment"] = {"python": sys.version, "platform": platform.platform()}
    raw = (ROOT / "requirements-lock.txt").read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode()
    lock = dict(l.strip().split("==") for l in text.splitlines() if "==" in l)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    have = dict(l.strip().split("==") for l in frozen.splitlines() if "==" in l)
    norm = lambda d: {k.lower().replace("_", "-"): v for k, v in d.items()}  # noqa: E731
    lock, have = norm(lock), norm(have)
    mism = {k: (v, have.get(k)) for k, v in lock.items() if have.get(k) != v}
    RECEIPT["environment"]["lock_vs_installed_mismatch"] = mism
    check("installed packages match requirements-lock.txt", not mism, f"{len(mism)} mismatches")
    for k, (want, got) in list(mism.items())[:15]:
        say(f"    {k}: lock={want} installed={got}")


# ---------------------------------------------------------------- 2. hashes (amendment, correction 2)
def hash_check():
    say("== 2. full SHA-256 values (amendment) and fail-closed data check")
    pinned = json.loads(check_data.PINNED_HASHES.read_text())["sha256"]
    check("pinned file has three full 64-hex digests", len(pinned) == 3 and all(
        re.fullmatch(r"[0-9a-f]{64}", v) for v in pinned.values()))
    v1 = json.loads((ROOT / "preflight/preflight_receipt.json").read_text())["data_sha256_full"]
    check("pinned digests equal those captured by preflight-v1 receipt", pinned == v1)
    proto = git("show", "protocol-v1:PROTOCOL.md")
    prefixes = {n: h.lower() for n, h in re.findall(r"(elliptic_txs_\w+\.csv)\s*=\s*([0-9A-Fa-f]+)", proto)}
    check("each protocol-v1 truncated hash is a prefix of the pinned full digest",
          set(prefixes) == set(pinned) and all(pinned[n].startswith(p) for n, p in prefixes.items()))
    actual = {n: check_data.sha256_file(C.DATA_DIR / n) for n in pinned if (C.DATA_DIR / n).exists()}
    check("data files on disk match the pinned full digests", actual == pinned)
    RECEIPT["data_sha256_full"] = actual

    # negative tests: the check must fail closed
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.json"
        wrong = dict(pinned)
        first = next(iter(wrong))
        wrong[first] = "0" * 64
        bad.write_text(json.dumps({"sha256": wrong}))
        check("verify_hashes exits non-zero on a wrong digest",
              exits_closed(check_data.verify_hashes, pinned_path=bad))
        check("verify_hashes exits non-zero on a missing file",
              exits_closed(check_data.verify_hashes, data_dir=Path(td) / "nowhere"))


# ---------------------------------------------------------------- 3. check_data path, audit not overwritten (correction 4)
def audit_check():
    say("== 3. check_data runs end to end without touching the tracked audit")
    tracked = ROOT / "results/raw/data_audit.csv"
    before = tracked.read_bytes()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            check_data.main()
        ok = True
    except SystemExit as e:
        ok = False
        say(f"    check_data exited: {e.code}")
    check("check_data.main() completes (hashes verified, audit equals committed)", ok)
    check("tracked results/raw/data_audit.csv bytes unchanged by check_data", tracked.read_bytes() == before)
    check("no data_audit_regenerated.csv was produced", not (ROOT / "results/raw/data_audit_regenerated.csv").exists())
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"
        raw.mkdir()
        altered = before + b"50,1,1,1,1\n"                                # one extra row: must not match
        (raw / "data_audit.csv").write_bytes(altered)
        old_raw = C.RAW_DIR
        C.RAW_DIR = raw
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                closed = exits_closed(check_data.main)
        finally:
            C.RAW_DIR = old_raw
        check("check_data fails closed on an audit mismatch", closed)
        check("on mismatch the committed file is untouched and a separate regenerated copy is written",
              (raw / "data_audit_regenerated.csv").exists() and (raw / "data_audit.csv").read_bytes() == altered)
    check("tracked audit still unchanged after the negative test", tracked.read_bytes() == before)


# ---------------------------------------------------------------- 4. data, splits, subgroups
def data_split_check():
    say("== 4. data counts, temporal splits, subgroups (structure only)")
    g = load_full()
    exp = dict(nodes=203769, edges=234355, illicit=4545, licit=42019, unknown=157205)
    got = dict(nodes=g.n, edges=int(g.edge_index.shape[1]), illicit=int((g.y == 1).sum()),
               licit=int((g.y == 0).sum()), unknown=int((g.y == -1).sum()))
    check("full-data counts match protocol section 3", got == exp, f"got={got}")
    check("0 edges cross time steps", int((g.time[g.edge_index[0]] != g.time[g.edge_index[1]]).sum()) == 0)
    exp_lab = {"train": (2219, 21387), "val": (1243, 5045), "test": (1083, 15587)}
    for fs in C.FEATURE_SETS:
        _, sp = build_splits(fs)
        for name, gr in sp.items():
            if fs == C.PRIMARY_FS:
                lab = (int((gr.y == 1).sum()), int((gr.y == 0).sum()))
                check(f"[{fs}] {name} labeled illicit/licit match section 4", lab == exp_lab[name], f"got={lab}")
            check(f"[{fs}] {name} edge indices valid", gr.edge_index.max(initial=-1) < gr.n)
        check(f"[{fs}] splits cover the stated time steps",
              set(sp["train"].time) == set(C.TRAIN_STEPS) and set(sp["val"].time) == set(C.VAL_STEPS)
              and set(sp["test"].time) == set(C.TEST_STEPS))
        check(f"[{fs}] node ids disjoint across splits",
              not (set(sp["train"].node_idx) & set(sp["val"].node_idx))
              and not (set(sp["val"].node_idx) & set(sp["test"].node_idx)))
        check(f"[{fs}] train features standardized on train period",
              abs(float(sp["train"].x.mean())) < 1e-3 and abs(float(sp["train"].x.std()) - 1) < 1e-2)
    _, sp = build_splits(C.PRIMARY_FS)
    tr, te = sp["train"], sp["test"]
    sizes = {}
    for name, gr in sp.items():
        grp = assign_group(gr.in_degree())[gr.y >= 0]
        yl = gr.y[gr.y >= 0]
        sizes[name] = {C.GROUP_LABELS[k]: (int((grp == k).sum()), int(((grp == k) & (yl == 1)).sum()))
                       for k in range(3)}
    RECEIPT["group_sizes_nodes_illicit"] = sizes
    check("train subgroup sizes match protocol section 9",
          sizes["train"] == {"in0": (7405, 582), "in1": (10710, 1559), "in2plus": (5491, 78)}, f"got={sizes['train']}")
    check("in_degree() equals in-degree only",
          bool((tr.in_degree() == np.bincount(tr.edge_index[1], minlength=tr.n)).all()))
    doc = (ROOT / "fraudshift/data.py").read_text()
    check("in_degree() docstring no longer claims total degree", "Total degree" not in doc)
    for wname, (a, b) in C.TEMPORAL_WINDOWS.items():
        n = int(((te.time >= a) & (te.time <= b) & (te.y == 1)).sum())
        say(f"    window {wname}: illicit test nodes = {n} (underpowered if < {C.MIN_ILLICIT})")
        RECEIPT.setdefault("window_illicit_counts", {})[wname] = n


# ---------------------------------------------------------------- 5. temporal path vs protocol-v1 and float64 reference
def old_vs_new_check():
    say("== 5. build_splits(): structure identical to protocol-v1, scaler accurate (correction 5)")
    src = git("show", "protocol-v1:fraudshift/data.py", strip=False)
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "oldpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        shutil.copy(ROOT / "fraudshift/config.py", pkg / "config.py")
        (pkg / "data.py").write_text(src)
        sys.path.insert(0, td)
        try:
            old = importlib.import_module("oldpkg.data")
            for fs in C.FEATURE_SETS:
                _, a = old.build_splits(fs)
                _, b = build_splits(fs)
                _, raw_sp = build_raw_splits(fs)
                same = all(np.array_equal(getattr(a[k], f), getattr(b[k], f))
                           for k in a for f in ("y", "edge_index", "time", "node_idx"))
                check(f"[{fs}] labels, edges, time steps, node ids identical to protocol-v1 build_splits", same)
                # float64 reference standardization from the raw train-period rows
                tr64 = raw_sp["train"].x.astype(np.float64)
                m64, s64 = tr64.mean(0), tr64.std(0)
                s64[s64 == 0] = 1.0
                worst_new = worst_old = worst_gap = 0.0
                for k in b:
                    ref = ((raw_sp[k].x.astype(np.float64) - m64) / s64)
                    worst_new = max(worst_new, float(np.abs(b[k].x - ref).max()))
                    worst_old = max(worst_old, float(np.abs(a[k].x - ref).max()))
                    worst_gap = max(worst_gap, float(np.abs(b[k].x - a[k].x).max()))
                check(f"[{fs}] standardized features equal the float64 reference within 1e-4", worst_new < 1e-4,
                      f"max abs diff {worst_new:.2e}")
                say(f"    [{fs}] protocol-v1 float32 scaler: max abs diff to float64 reference {worst_old:.2e}; "
                    f"new vs protocol-v1: {worst_gap:.2e}")
                RECEIPT.setdefault("scaler_precision", {})[fs] = dict(
                    new_vs_float64=worst_new, protocol_v1_vs_float64=worst_old, new_vs_protocol_v1=worst_gap)
                sds = raw_sp["train"].x.std(0, dtype=np.float64)
                check(f"[{fs}] no zero or near-constant feature column in the train period (min sd > 1e-6)",
                      float(sds.min()) > 1e-6, f"min sd {float(sds.min()):.3e}")
        finally:
            sys.path.remove(td)
            for m in [m for m in sys.modules if m.startswith("oldpkg")]:
                del sys.modules[m]


# ---------------------------------------------------------------- 6. strict inductive preprocessing (correction 3)
def inductive_check():
    say("== 6. strict inductive split: scaler fit once on retained nodes, independent of held-out features")
    _, raw = build_raw_splits(C.PRIMARY_FS)
    raw_tr, raw_val = raw["train"], raw["val"]
    _, std = build_splits(C.PRIMARY_FS)
    ok_leak = ok_scaler = ok_indep = ok_pert = ok_same_as_v1 = True
    worst = worst_v1 = 0.0
    for seed in C.SEEDS:
        sp = make_inductive_split(raw_tr, raw_val, seed)      # also runs the in-code leakage asserts
        mask = np.zeros(raw_tr.n, dtype=bool)
        mask[sp.held] = True
        ok_leak &= not np.isin(sp.g_train.node_idx, raw_tr.node_idx[sp.held]).any()
        # scaler equals one fit directly on the retained raw rows, nothing else
        mu, sd = fit_scaler(raw_tr.x[~mask])
        ok_scaler &= np.array_equal(sp.scaler[0], mu) and np.array_equal(sp.scaler[1], sd)
        # change held-out feature values: scaler and training features must not move at all
        x2 = raw_tr.x.copy()
        x2[sp.held] = x2[sp.held] * 1000.0 + 7.0
        raw2 = Graph(x2, raw_tr.y, raw_tr.edge_index, raw_tr.time, raw_tr.node_idx)
        sp2 = make_inductive_split(raw2, raw_val, seed)
        ok_indep &= (np.array_equal(sp.scaler[0], sp2.scaler[0]) and np.array_equal(sp.scaler[1], sp2.scaler[1])
                     and np.array_equal(sp.g_train.x, sp2.g_train.x) and np.array_equal(sp.g_val.x, sp2.g_val.x))
        ok_pert &= not np.array_equal(sp.g_infer.x[sp.held], sp2.g_infer.x[sp.held])   # perturbation was real
        # final training features equal a float64 reference computed from the retained raw rows only
        r64 = raw_tr.x[~mask].astype(np.float64)
        s64 = r64.std(0)
        s64[s64 == 0] = 1.0
        d = float(np.abs(sp.g_train.x - (r64 - r64.mean(0)) / s64).max())
        worst = max(worst, d)
        ok_same_as_v1 &= d < 1e-4
        # informational: distance to the protocol-v1 two-step float32 route
        xs = std["train"].x[~mask]
        m1, s1 = xs.mean(0), xs.std(0)
        s1[s1 == 0] = 1.0
        worst_v1 = max(worst_v1, float(np.abs(((xs - m1) / s1) - sp.g_train.x).max()))
    check("inductive: held-out nodes and all incident edges absent from training graph, seeds 0-9", ok_leak)
    check("inductive: scaler equals a fit on retained raw rows only, seeds 0-9", ok_scaler)
    check("inductive: changing held-out feature values cannot change the retained-node scaler, "
          "training features or validation features (bitwise), seeds 0-9", ok_indep)
    check("inductive: the held-out perturbation was real (inference features did change)", ok_pert)
    check("inductive: final training features equal a float64 reference from retained raw rows (within 1e-4), "
          "seeds 0-9", ok_same_as_v1, f"max abs difference {worst:.2e}")
    say(f"    informational: distance to the protocol-v1 two-step float32 route {worst_v1:.2e}")
    RECEIPT["inductive_vs_float64_ref_max_abs"] = worst
    RECEIPT["inductive_vs_protocol_v1_route_max_abs"] = worst_v1
    src = (ROOT / "fraudshift/inductive_split.py").read_text()
    check("inductive_split.py contains no model code", "models" not in src.replace("model list", ""))


# ---------------------------------------------------------------- 7. frozen files, boundary, outcome files
def boundary_check():
    say("== 7. frozen files, allowed changes, fit boundary, outcome files untouched")
    for f in FROZEN_FILES:
        check(f"{f} byte-identical to protocol-v1", git("diff", "protocol-v1", "--", f) == "")
    changed = set(git("diff", "--name-only", "protocol-v1").splitlines())
    changed |= set(git("ls-files", "--others", "--exclude-standard").splitlines())
    stray = sorted(p for p in changed if p not in ALLOWED_CHANGED and not p.startswith(ALLOWED_PREFIXES))
    check("only allowed files differ from protocol-v1", not stray, f"unexpected: {stray}")
    RECEIPT["changed_vs_protocol_v1"] = sorted(changed)
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith("preflight/")]
    check("working tree clean except preflight/ outputs (receipt belongs to committed code)", not dirty, str(dirty))
    sh = (ROOT / "run_all.sh").read_text()
    order = re.findall(r"python -m fraudshift\.(\w+)", sh)
    check("run_all.sh order unchanged", order == EXPECTED_ORDER, str(order))
    for m in order:
        check(f"module {m} compiles", subprocess.run(
            [sys.executable, "-m", "py_compile", str(ROOT / f"fraudshift/{m}.py")]).returncode == 0)
    import ast
    tree = ast.parse((ROOT / "fraudshift/run_seeds.py").read_text())
    imp = any(isinstance(n, ast.Import) and any(a.name == "time" for a in n.names) for n in tree.body)
    shadow = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
              for x in ast.walk(t) if isinstance(x, ast.Name) and x.id == "time"]
    check("run_seeds.py does not overwrite the imported `time` module", not (imp and shadow), f"lines {shadow}")
    check("first fit happens in run_seeds, which is not executed here", True)
    recorded = (ROOT / "amendments/AMENDMENT_1_code.diff").read_text().replace("\r\n", "\n")
    now = git("diff", "protocol-v1", "--", "fraudshift", strip=False).replace("\r\n", "\n")
    strip_idx = lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("index "))  # noqa: E731
    check("amendments/AMENDMENT_1_code.diff equals git diff protocol-v1 -- fraudshift",
          strip_idx(recorded) == strip_idx(now))
    results = ROOT / "results"
    present = []
    for d in ("scores", "inductive", "plots"):
        p = results / d
        if p.exists() and any(p.iterdir()):
            present.append(f"results/{d}/")
    rawdir = results / "raw"
    present += [f"results/raw/{f.name}" for f in rawdir.iterdir() if f.name in OUTCOME_CSVS or
                f.name == "data_audit_regenerated.csv"]
    check("no outcome-bearing files exist (listed by name only, none opened)", not present, str(present))
    RECEIPT["results_raw_listing"] = sorted(f.name for f in rawdir.iterdir())
    RECEIPT["git"] = {"head": git("rev-parse", "HEAD"), "protocol_tag": git("rev-parse", "protocol-v1^{commit}"),
                      "preflight_v1_commit": "f7d016b2fb7e5ec12b9d8b6e5db3af3683ec3f44"}
    RECEIPT["amendment_sha256"] = hashlib.sha256((ROOT / "amendments/AMENDMENT_1.md").read_bytes()).hexdigest()


def main():
    for step in (env_check, hash_check, audit_check, data_split_check, old_vs_new_check, inductive_check,
                 boundary_check):
        try:
            step()
        except Exception as e:  # a crash is itself a finding
            check(f"{step.__name__} ran without exception", False, repr(e))
    RECEIPT["no_outcome_statement"] = ("No model was fit and no metric, loss or prediction was produced; "
                                       "no outcome-bearing file was opened.")
    out = ROOT / "preflight"
    out.mkdir(exist_ok=True)
    (out / "preflight_v2_receipt.json").write_text(json.dumps(RECEIPT, indent=2, default=str))
    (out / "preflight_v2_log.txt").write_text("\n".join(LOG) + "\n")
    failed = [k for k, v in RECEIPT.get("checks", {}).items() if not v["pass"]]
    say(f"\n{len(failed)} failing checks" if failed else "\nall checks passed")


if __name__ == "__main__":
    main()
