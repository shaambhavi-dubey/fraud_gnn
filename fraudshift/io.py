import json

import numpy as np

from . import config as C


def iter_scores(fs=None, directory=None):
    """Yield one dict per saved (model, feature set, seed) test-score file."""
    directory = directory or C.SCORES_DIR
    for f in sorted(directory.glob("*.npz")):
        d = np.load(f)
        if fs is not None and str(d["fs"]) != fs:
            continue
        yield dict(model=str(d["model"]), fs=str(d["fs"]), seed=int(d["seed"]),
                   thr=float(d["thr"]), y=d["y"], s=d["score"], t=d["time"], indeg=d["indeg"])



