"""Step 1: audit the real data BEFORE freezing the protocol. Uses no model outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import assign_group, build_splits, load_full

# Full 64-character SHA-256 values, appended by amendments/AMENDMENT_1.md
# (protocol-v1 section 3 lists only truncated prefixes and is left unchanged).
PINNED_HASHES = Path(__file__).resolve().parent.parent / "amendments" / "pinned_data_hashes.json"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_hashes(pinned_path=None, data_dir=None):
    """Fail closed: stop unless every pinned data file matches its full SHA-256."""
    pinned = json.loads(Path(pinned_path or PINNED_HASHES).read_text())["sha256"]
    data_dir = Path(data_dir or C.DATA_DIR)
    for name, want in pinned.items():
        f = data_dir / name
        if not f.exists():
            raise SystemExit(f"FAIL: pinned data file missing: {f}")
        got = sha256_file(f)
        print(f"sha256 {name}: {'OK' if got == want.lower() else 'MISMATCH'}")
        if got != want.lower():
            raise SystemExit(f"FAIL: SHA-256 mismatch for {name}\n  pinned: {want}\n  actual: {got}")


def main():
    verify_hashes()
    g = load_full()
    n_edges = g.edge_index.shape[1]
    print(f"nodes            : {g.n}")
    print(f"edges            : {n_edges}")
    print(f"feature columns  : {g.x.shape[1]}  (expect 165)")
    print(f"time steps       : {g.time.min()}..{g.time.max()}  ({len(np.unique(g.time))} unique, expect 49)")
    print(f"illicit / licit / unknown: {(g.y==1).sum()} / {(g.y==0).sum()} / {(g.y==-1).sum()}")
    print(f"illicit share of labeled : {(g.y==1).sum() / (g.y>=0).sum():.4f}")

    cross = int((g.time[g.edge_index[0]] != g.time[g.edge_index[1]]).sum())
    print(f"edges crossing time steps: {cross}  (expect 0)")
    assert cross == 0, "Edges cross time steps: the split assumptions in PROTOCOL.md are wrong."

    rows = []
    for t in sorted(np.unique(g.time)):
        m = g.time == t
        rows.append(dict(time_step=int(t), nodes=int(m.sum()), illicit=int((g.y[m] == 1).sum()),
                         licit=int((g.y[m] == 0).sum()), unknown=int((g.y[m] == -1).sum())))
    df = pd.DataFrame(rows)
    # The committed audit is a tracked artifact: compare, never overwrite (fail closed).
    committed_path = C.RAW_DIR / "data_audit.csv"
    if not committed_path.exists():
        raise SystemExit(f"FAIL: committed audit not found: {committed_path}")
    if not df.equals(pd.read_csv(committed_path)):
        regen = C.RAW_DIR / "data_audit_regenerated.csv"
        df.to_csv(regen, index=False)
        raise SystemExit(f"FAIL: recomputed audit differs from {committed_path}; "
                         f"regenerated copy written to {regen}, tracked file untouched")
    print("\nper time step: recomputed audit equals committed results/raw/data_audit.csv")
    print(df.to_string(index=False))

    _, sp = build_splits(C.PRIMARY_FS)
    print()
    for name, gr in sp.items():
        print(f"{name:5s}: nodes={gr.n:7d} edges={gr.edge_index.shape[1]:7d} "
              f"illicit={int((gr.y==1).sum()):5d} licit={int((gr.y==0).sum()):6d}")
    print(f"\nin-degree groups (fixed bins): {list(C.GROUP_LABELS.values())}")
    for name in ("train", "val", "test"):
        gr = sp[name]
        grp = assign_group(gr.in_degree())[gr.y >= 0]
        yl = gr.y[gr.y >= 0]
        print(f"{name:5s} labeled group sizes: {[int((grp == k).sum()) for k in range(3)]}  "
              f"illicit: {[int(((grp == k) & (yl == 1)).sum()) for k in range(3)]}")


if __name__ == "__main__":
    main()
