"""Pre-outcome reproducibility preflight for the fraud-GNN study.

Run from the repo root:   python preflight/run_preflight.py
Writes: preflight/preflight_receipt.json and preflight/preflight_log.txt

Boundary: this script NEVER fits a model and NEVER reads results/scores. It does not
import fraudshift.models, run_seeds, shifts, inductive, stats or plots. It reports
counts, hashes, versions and pass/fail only. No metric, loss, prediction or
class-conditional score is computed or written.
It does not call check_data.main(), because that overwrites the tracked
results/raw/data_audit.csv; the audit is recomputed in memory and compared instead.
"""
import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraudshift import config as C  # noqa: E402
from fraudshift.data import assign_group, build_splits, load_full  # noqa: E402

LOG, RECEIPT = [], {}


def say(msg):
    print(msg)
    LOG.append(msg)


def check(name, ok, detail=""):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    RECEIPT.setdefault("checks", {})[name] = {"pass": bool(ok), "detail": str(detail)}
    return ok


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- 1. environment
def env_check():
    say("== 1. environment and dependency lock")
    RECEIPT["environment"] = {"python": sys.version, "platform": platform.platform()}
    raw = (ROOT / "requirements-lock.txt").read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode()
    RECEIPT["environment"]["lock_encoding"] = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    lock = dict(l.strip().split("==") for l in text.splitlines() if "==" in l)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    have = dict(l.strip().split("==") for l in frozen.splitlines() if "==" in l)
    norm = lambda d: {k.lower().replace("_", "-"): v for k, v in d.items()}
    lock, have = norm(lock), norm(have)
    mism = {k: (v, have.get(k)) for k, v in lock.items() if have.get(k) != v}
    RECEIPT["environment"]["lock_vs_installed_mismatch"] = mism
    check("installed packages match requirements-lock.txt", not mism, f"{len(mism)} mismatches")
    for k, (want, got) in list(mism.items())[:15]:
        say(f"    {k}: lock={want} installed={got}")


# ---------------------------------------------------------------- 2. data hashes
def hash_check():
    say("== 2. data hashes (protocol section 3)")
    proto = (ROOT / "PROTOCOL.md").read_text()
    pinned = {n: h.lower() for n, h in re.findall(r"(elliptic_txs_\w+\.csv)\s*=\s*([0-9A-Fa-f]+)", proto)}
    out = {}
    for name, ph in pinned.items():
        f = C.DATA_DIR / name
        if not f.exists():
            check(f"file present: {name}", False, str(f))
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        out[name] = h
        check(f"{name} digest starts with protocol value", h.startswith(ph), f"protocol length={len(ph)} hex chars")
        check(f"{name} protocol value is a full 64-char SHA-256", len(ph) == 64, f"length={len(ph)}")
    RECEIPT["data_sha256_full"] = out


# ---------------------------------------------------------------- 3. data + splits
def data_split_check():
    say("== 3. data counts, splits, subgroups (structure only)")
    g = load_full()
    cross = int((g.time[g.edge_index[0]] != g.time[g.edge_index[1]]).sum())
    exp = dict(nodes=203769, edges=234355, illicit=4545, licit=42019, unknown=157205)
    got = dict(nodes=g.n, edges=int(g.edge_index.shape[1]), illicit=int((g.y == 1).sum()),
               licit=int((g.y == 0).sum()), unknown=int((g.y == -1).sum()))
    check("full-data counts match protocol section 3", got == exp, f"got={got}")
    check("0 edges cross time steps", cross == 0, f"cross={cross}")
    check("49 time steps", len(np.unique(g.time)) == 49)

    rows = []
    for t in sorted(np.unique(g.time)):
        m = g.time == t
        rows.append(dict(time_step=int(t), nodes=int(m.sum()), illicit=int((g.y[m] == 1).sum()),
                         licit=int((g.y[m] == 0).sum()), unknown=int((g.y[m] == -1).sum())))
    committed = pd.read_csv(ROOT / "results/raw/data_audit.csv")
    check("recomputed audit equals committed results/raw/data_audit.csv",
          pd.DataFrame(rows).equals(committed))

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
    exp_g = {"in0": (7405, 582), "in1": (10710, 1559), "in2plus": (5491, 78)}
    check("train subgroup sizes match protocol section 9", sizes["train"] == exp_g, f"got={sizes['train']}")
    check("in_degree() equals in-degree only (not total degree)",
          bool((tr.in_degree() == np.bincount(tr.edge_index[1], minlength=tr.n)).all()))
    for wname, (a, b) in C.TEMPORAL_WINDOWS.items():
        m = (te.time >= a) & (te.time <= b) & (te.y == 1)
        say(f"    window {wname}: illicit test nodes = {int(m.sum())} (underpowered if < {C.MIN_ILLICIT})")
        RECEIPT.setdefault("window_illicit_counts", {})[wname] = int(m.sum())


# ---------------------------------------------------------------- 4. inductive split construction
def inductive_check():
    say("== 4. strict inductive split construction (structure only, no fit)")
    src = (ROOT / "fraudshift/inductive.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "stratified_holdout")
    ns = {"np": np}
    exec(ast.get_source_segment(src, fn), ns)  # same source text, without importing torch code
    _, sp = build_splits(C.PRIMARY_FS)
    gt = sp["train"]
    ok = True
    for seed in C.SEEDS:
        rng = np.random.default_rng(1000 + seed)
        held = ns["stratified_holdout"](rng, gt.y, C.INDUCTIVE_HOLDOUT)
        mask = np.zeros(gt.n, dtype=bool)
        mask[held] = True
        g_tr = gt.subset(~mask)
        s, d = gt.edge_index
        ok &= not np.isin(g_tr.node_idx, gt.node_idx[held]).any()
        ok &= g_tr.n == gt.n - len(held)
        ok &= bool((gt.y[held] >= 0).all())
        ok &= g_tr.edge_index.max(initial=-1) < g_tr.n
        # every edge touching a held-out node is absent from the training graph
        incident = int((mask[s] | mask[d]).sum())
        ok &= (g_tr.edge_index.shape[1] == gt.edge_index.shape[1] - incident)
    check("inductive: held-out nodes and all incident edges absent from training graph, seeds 0-9", ok)


# ---------------------------------------------------------------- 5. fit boundary + frozen files
def boundary_check():
    say("== 5. one-command path up to the model-fit boundary")
    sh = (ROOT / "run_all.sh").read_text()
    mods = re.findall(r"python -m fraudshift\.(\w+)", sh)
    RECEIPT["run_all_order"] = mods
    say(f"    run_all.sh order: {mods}")
    for m in mods:
        check(f"module {m} exists and compiles", subprocess.run(
            [sys.executable, "-m", "py_compile", str(ROOT / f"fraudshift/{m}.py")]).returncode == 0)
    say("    first fit happens in: run_seeds (not executed here)")
    tree = ast.parse((ROOT / "fraudshift/run_seeds.py").read_text())
    imported_time = any(isinstance(n, ast.Import) and any(a.name == "time" for a in n.names) for n in tree.body)
    shadow = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign)
              for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name) and x.id == "time"]
    check("run_seeds.py does not overwrite the imported `time` module", not (imported_time and shadow),
          f"assignment to name `time` at line(s) {shadow}")
    check("PROTOCOL.md unchanged vs tag protocol-v1", git("diff", "protocol-v1", "--", "PROTOCOL.md") == "")
    check("fraudshift/config.py unchanged vs tag protocol-v1", git("diff", "protocol-v1", "--", "fraudshift/config.py") == "")
    RECEIPT["git"] = {"head": git("rev-parse", "HEAD"), "protocol_tag": git("rev-parse", "protocol-v1^{commit}")}


def main():
    for step in (env_check, hash_check, data_split_check, inductive_check, boundary_check):
        try:
            step()
        except Exception as e:  # a crash is itself a finding
            check(f"{step.__name__} ran without exception", False, repr(e))
    RECEIPT["no_outcome_statement"] = "No model was fit and no metric, loss or prediction was produced."
    out = ROOT / "preflight"
    out.mkdir(exist_ok=True)
    (out / "preflight_receipt.json").write_text(json.dumps(RECEIPT, indent=2, default=str))
    (out / "preflight_log.txt").write_text("\n".join(LOG) + "\n")
    failed = [k for k, v in RECEIPT.get("checks", {}).items() if not v["pass"]]
    say(f"\n{len(failed)} failing checks" if failed else "\nall checks passed")


if __name__ == "__main__":
    main()