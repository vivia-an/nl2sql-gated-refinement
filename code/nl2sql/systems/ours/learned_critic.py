"""Learned critic V5 (E5): exported model bundle + loader used by m5.Critic (`m5.variant: v5`) and by the trainer
(`analysis/critic_v5.py`). The bundle is a pickle written by `save_bundle` (trusted local artefact, sha256 recorded by the
dev lock via `m5.learned.path`; same sklearn version as the venv that trained it):
  {"kind": "critic_v5", "model_name": "hgb"|"lr"|"lgbm", "feature_names": [...], "with_v4": bool, "model": fitted estimator,
   "calibrator": sklearn IsotonicRegression | None, "meta": {...training provenance...}}
Config (drop-in):   m5: {variant: v5, learned: {path: exp/frozen/critic_v5.pkl}}   (see exp/configs/crit_v5.yaml)
The live score is `10 * p_correct` (calibrated probability that the candidate is EX-correct), so the patent thresholds
theta_acc/theta_fb/theta_h keep their meaning on the 0-10 scale; `score_v3` keeps the original V3 score."""
import os
import pickle

import numpy as np

from ... import paths
from .critic_features import record_features, vectorize


def save_bundle(path, model, feature_names, model_name, calibrator=None, with_v4=False, meta=None):
    b = {"kind": "critic_v5", "model_name": model_name, "feature_names": list(feature_names), "with_v4": bool(with_v4),
         "model": model, "calibrator": calibrator, "meta": meta or {}}
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(b, f, protocol=4)
    os.replace(tmp, str(path))


class LearnedCritic:
    def __init__(self, bundle):
        assert bundle.get("kind") == "critic_v5", "not a critic_v5 bundle"
        self.b = bundle
        self.names = bundle["feature_names"]
        self.model = bundle["model"]
        self.cal = bundle.get("calibrator")
        self.with_v4 = bundle.get("with_v4", False)

    @classmethod
    def load(cls, path):
        p = str(path)
        if not os.path.isabs(p):
            p = str(paths.REPO / p)
        with open(p, "rb") as f:
            return cls(pickle.load(f))

    def predict_matrix(self, X):
        p = self.model.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]
        return self.cal.predict(p) if self.cal is not None else p

    def p_correct(self, rec):
        x = vectorize(record_features(rec, with_v4=self.with_v4), self.names)
        return float(self.predict_matrix([x])[0])
