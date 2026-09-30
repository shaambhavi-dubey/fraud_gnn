"""Compute receipt: environment and training time, written after the runs."""
import json
import os
import platform
import sys

import numpy as np

from . import config as C


def main():
    import matplotlib
    import pandas
    import scipy
    import sklearn
    import torch
    import torch_geometric
    import xgboost

    per_run = {}
    for f in sorted(C.SCORES_DIR.glob("*.npz")):
        d = np.load(f)
        if "seconds" not in d.files:
            continue
        key = str(d["model"]) + "/" + str(d["fs"])
        per_run.setdefault(key, []).append(float(d["seconds"]))

    receipt = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "libraries": {
            "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
            "xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "training_seconds": {k: {"runs": len(v), "total": round(sum(v), 1),
                                 "mean": round(sum(v) / len(v), 1)} for k, v in per_run.items()},
        "total_training_seconds": round(sum(sum(v) for v in per_run.values()), 1),
        "note": "Covers model training and scoring in run_seeds only. Shifts, inductive, "
                "stats and plots are not timed here. Working hours are in hours_log.md.",
    }
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RAW_DIR / "compute_receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()