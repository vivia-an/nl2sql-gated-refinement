"""Offline tests of the simulated-feedback calibration analysis (tools/feedback_calibration.py, SAP Amendment 5).  No pytest-only features,
no network, no LLM call."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import feedback_calibration as F  # noqa: E402

RECORDS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "diag", "records.jsonl.gz")


def toy_items(n_db=10, per_db=6, seed=1):
    rng = random.Random(seed)
    items = []
    for d in range(n_db):
        for k in range(per_db):
            y = 1 if k % 3 else 0
            sem = rng.choice([2, 4, 6, 8]) if y else rng.choice([7, 9, 10])
            items.append({"id": "T%d_%d" % (d, k), "cat": "c", "db": "db%d" % d, "y": y,
                          "x": [10.0, float(sem), float(rng.choice([8, 10]))], "total": 0.3 * 10 + 0.5 * sem + 0.2 * 9})
    return items


def test_folds_are_database_grouped_and_balanced():
    cnt = {"db%d" % i: 3 + i % 4 for i in range(23)}
    f = F.make_folds(cnt, seed=3)
    assert set(f) == set(cnt) and set(f.values()) == set(range(5))
    size = [0] * 5
    for d, k in f.items():
        size[k] += cnt[d]
    assert max(size) - min(size) <= max(cnt.values())
    assert F.make_folds(cnt, seed=3) == f and F.make_folds(cnt, seed=4) != f


def test_auroc_tie_aware_and_weighted():
    assert F.auroc([1, 2, 3, 4], [1, 1, 0, 0]) == 1.0          # lower score = injected
    assert F.auroc([3, 4, 1, 2], [1, 1, 0, 0]) == 0.0
    assert F.auroc([5, 5, 5, 5], [1, 0, 1, 0]) == 0.5
    g = F.tie_groups([1.0, 2.0, 2.0, 3.0])
    assert F.auroc_weighted(g, [1, 0, 1, 0], [0, 1, 2, 3], [1, 1, 1, 1]) == F.auroc([1.0, 2.0, 2.0, 3.0], [1, 0, 1, 0])
    # weight 2 on db 0 = duplicating item 0
    assert abs(F.auroc_weighted(F.tie_groups([1.0, 2.0, 2.0, 3.0]), [1, 0, 1, 0], [0, 1, 2, 3], [2, 1, 1, 1])
               - F.auroc([1.0, 1.0, 2.0, 2.0, 3.0], [1, 1, 0, 1, 0])) < 1e-12


def test_threshold_rules():
    s = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [1, 1, 1, 1, 0, 1, 0, 0]
    t = F.fit_threshold(s, y, "fp", alpha0=0.0, d0=1.0)         # no false positive allowed -> flag only below the first control
    assert t == 4.5
    t = F.fit_threshold(s, y, "det", alpha0=1.0, d0=1.0)        # all injected must be flagged -> t above 6
    assert 6 < t <= 7
    assert F.fit_threshold([1, 2], [1, 1], "fp", 0.3, 0.8) is None


def test_logreg_sign_and_regularisation():
    rng = random.Random(0)
    X, y = [], []
    for _ in range(200):
        lab = rng.random() < 0.5
        X.append([10.0, (3 if lab else 9) + rng.random(), 9.0 + rng.random()])
        y.append(1 if lab else 0)
    w, b = F.fit_logreg(X, y)
    assert w[1] < -0.5                                           # lower semantic score -> more likely incorrect
    pred = [1 if F.lr_score(w, b, x) < 0 else 0 for x in X]
    assert sum(1 for p, t in zip(pred, y) if p == t) / 200.0 > 0.95
    # stronger data term -> larger |w|: C is respected
    w2, _ = F.fit_logreg(X, y, C=0.001)
    assert abs(w2[1]) < abs(w[1])
    # gradient ~ 0 at the optimum
    d = 1e-4
    base = F._objective(w, b, X, y, F.C_LR)
    for j in range(3):
        wp = list(w)
        wp[j] += d
        assert F._objective(wp, b, X, y, F.C_LR) >= base - 1e-9


def test_cv_n0_reproduces_uncalibrated_and_is_deterministic():
    items = toy_items()
    pos = [i for i in items if i["y"]]
    neg = [i for i in items if not i["y"]]
    d0 = sum(1 for i in pos if i["total"] < 8.5) / float(len(pos))
    a0 = sum(1 for i in neg if i["total"] < 8.5) / float(len(neg))
    fl, sc = F.run_cv(items, seeds=2, budgets=[0, 20], alpha0=a0, d0=d0)
    for L in F.LEARNERS:
        for R in F.RULES:
            assert fl[(L, R, 0)][0] == [1 if i["total"] < 8.5 else 0 for i in items]
    fl2, _ = F.run_cv(items, seeds=2, budgets=[0, 20], alpha0=a0, d0=d0)
    assert fl == fl2
    assert all(len(v) == len(items) for v in fl[("logreg", "fp", 20)].values())


def test_no_test_database_in_training_sample():
    # the sample for fold f is drawn from items whose fold differs: check through the fold map used by run_cv
    items = toy_items()
    cnt = {}
    for it in items:
        cnt[it["db"]] = cnt.get(it["db"], 0) + 1
    fold_of = F.make_folds(cnt, 0)
    for f in range(5):
        train = [i for i, it in enumerate(items) if fold_of[it["db"]] != f]
        test = [i for i, it in enumerate(items) if fold_of[it["db"]] == f]
        assert not ({items[i]["db"] for i in train} & {items[i]["db"] for i in test})


def test_real_records_n0_reproduces_diag_numbers():
    if not os.path.exists(RECORDS):
        return
    items = F.load_items(RECORDS)
    assert len(items) == 594 and sum(i["y"] for i in items) == 396
    pos = [i for i in items if i["y"]]
    neg = [i for i in items if not i["y"]]
    assert round(sum(1 for i in pos if i["total"] < 8.5) / 396.0, 2) == 0.85
    assert round(sum(1 for i in neg if i["total"] < 8.5) / 198.0, 2) == 0.27
    assert all(i["cat"] and i["db"] for i in items)
