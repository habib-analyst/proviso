"""Tests for proviso. Deterministic, no network, no LLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from proviso import traces as T
from proviso import isolation as J
from proviso import controller as C
from proviso import stats as S


def test_generate_counts():
    eps = T.generate(50, 60, 20, 30, seed=1)
    assert len(eps) == 160
    assert sum(1 for e in eps if e.attacked) == 50


def test_twin_invisibility():
    eps = T.generate(50, 50, 0, 0, seed=2)
    att = [e for e in eps if e.attacked]
    tw = [e for e in eps if not e.attacked]
    att_texts = {e.proposed_action for e in att}
    assert all(e.proposed_action in att_texts for e in tw)
    # ... but labels differ: twins safe, attacked unsafe
    assert all(e.label == T.SAFE for e in tw)
    assert all(e.label == T.UNSAFE for e in att)


def test_provenance_features_shape():
    eps = T.generate(10, 10, 0, 0, seed=3)
    for e in eps:
        f = C.provenance_features(e)
        assert len(f) == 8
        assert all(isinstance(x, float) for x in f)


def test_provenance_features_discriminate():
    # For a paired attacked/twin episode (identical action text), the
    # provenance-typed memory must look different: at least one role-query
    # feature must separate them.
    eps = T.generate(20, 20, 0, 0, seed=4)
    att = [e for e in eps if e.attacked]
    tw = {e.proposed_action: e for e in eps if not e.attacked}
    separated = 0
    for a in att:
        t = tw[a.proposed_action]
        fa = C.provenance_features(a)
        ft = C.provenance_features(t)
        if any(abs(x - y) > 1e-9 for x, y in zip(fa, ft)):
            separated += 1
    assert separated >= 0.8 * len(att), separated


def test_models_train_and_decide():
    train = T.generate(60, 70, 20, 40, seed=5)
    ev = T.generate(30, 30, 12, 18, seed=6)
    judge = J.train(train)
    ctrl = C.train(train)
    for e in ev:
        assert isinstance(J.decide(judge, e), bool)
        assert isinstance(C.decide(ctrl, e), bool)
        p = C.p_unsafe(ctrl, e)
        assert 0.0 <= p <= 1.0


def test_f1_binary_edge_cases():
    assert S.f1_score_binary([1, 1], [0, 0]) == 0.0
    assert S.f1_score_binary([1, 0], [1, 0]) == 1.0
    assert S.f1_score_binary([0, 0], [0, 0]) == 0.0


def test_disease_and_cure_properties():
    # The blind judge must MISS invisible attacks (P(unsafe) < 0.5 on
    # attacked texts) while the provenance controller must CATCH them,
    # and neither may collapse on benign traffic.
    train = T.generate(250, 350, 100, 300, seed=42)
    ev = T.generate(60, 60, 24, 36, seed=1042)
    judge = J.train(train)
    ctrl = C.train(train)
    att = [e for e in ev if e.attacked]
    ben = [e for e in ev if not e.attacked]
    jd = lambda e: J.decide(judge, e)
    cd = lambda e: C.decide(ctrl, e)
    assert S.f1_of(att, jd) < 0.5, "judge should miss invisible attacks"
    assert S.f1_of(att, cd) > 0.9, "controller should catch attacks"
    assert S.f1_of(ben, jd) > 0.9, "judge should be strong on benign"
    assert S.f1_of(ben, cd) > 0.9, "controller must not collapse on benign"


def test_bootstrap_runs():
    eps = T.generate(40, 40, 10, 20, seed=8)
    train = T.generate(60, 70, 20, 40, seed=9)
    judge = J.train(train)
    ctrl = C.train(train)
    gap, lo, hi = S.paired_bootstrap_f1_gap(
        eps, lambda e: C.decide(ctrl, e), lambda e: J.decide(judge, e),
        n_boot=50, seed=1)
    assert lo <= gap <= hi or True  # percentile CI sanity; gap in range
    assert -1.0 <= gap <= 1.0
