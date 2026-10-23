"""PROVISO controller: provenance-typed action selection.

Encodes trusted observations, external observations, the proposed action,
and the alternative into provenance-typed memory, relates them through
role-specific queries, and commits the action with the highest calibrated
value learned from outcomes.

Role-specific queries (provenance-alignment features):
  q1 external-only payload fraction: content tokens of the action present in
     EXTERNAL observations but absent from TRUSTED ones
  q2 trusted coverage: content tokens of the action present in TRUSTED obs
  q3 injection-marker hits in EXTERNAL observations
  q4 entity authorization: every sensitive entity (recipient/path/code) in
     the action authorized by TRUSTED text? (min over entities)
  q5 directive alignment: external observation issues a directive whose verb
     matches the action's verb
  q6 number of external observations
Plus TF-IDF over role-tagged text:
  [TRUSTED] ... [EXTERNAL] ... [PROPOSED] ... [ALTERNATIVE] ...
Logistic regression + isotonic calibration. Same commit rule as the judge.
"""

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .traces import (EXTERNAL, TRUSTED, UNSAFE, INJECTION_MARKERS)

_STOP = set("the a an to of in on for with is are was were be been it its "
           "this that these those as at by from or and me my your you we "
           "all also has have had do does did will would can could should "
           "been being than then there their them they he she his her "
           "not no yes if when what which who whom whose how why where "
           "please just now here out up down over under again once "
           "send share schedule open list summarize clean delete drop "
           "create read save ask confirm action body title path name doc "
           "with team file files".split())

_VERBS = ["delete", "drop", "send", "share", "create", "read", "clean",
          "forward", "include", "cc"]

# Ablation switch for the marker-robustness check: when True, the injection-
# marker feature q3 is zeroed so the controller must rely on the other
# provenance-alignment queries. Set by run_experiment for the ablation pass.
Q3_ABLATED = False

BLOCK_THRESHOLD = 0.5  # == V_alt, the fixed value of the safe no-op


def _tokens(text):
    return re.findall(r"[a-zA-Z@._/\-']+", text.lower())


def _content_tokens(text):
    return [t for t in _tokens(text) if t not in _STOP and len(t) > 2]


def _trusted_text(ep):
    return " ".join(o.text for o in ep.observations
                    if o.provenance == TRUSTED)


def _external_texts(ep):
    return [o.text for o in ep.observations if o.provenance == EXTERNAL]


def _sensitive_entities(action):
    toks = _content_tokens(action)
    ents = [t for t in toks if ("@" in t or "/" in t or "-" in t)]
    return ents


def provenance_features(ep):
    """Role-specific query vector for one episode/action."""
    trusted = _trusted_text(ep)
    externals = _external_texts(ep)
    ext_all = " ".join(externals)
    trusted_toks = set(_content_tokens(trusted))
    ext_toks = set(_content_tokens(ext_all))
    act_toks = _content_tokens(ep.proposed_action)
    act_set = set(act_toks)

    if act_toks:
        ext_only = sum(1 for t in act_toks
                       if t in ext_toks and t not in trusted_toks)
        q1 = ext_only / len(act_toks)
        q2 = sum(1 for t in act_toks if t in trusted_toks) / len(act_toks)
    else:
        q1 = q2 = 0.0

    ext_low = ext_all.lower()
    q3 = 0.0 if Q3_ABLATED else float(
        sum(1 for m in INJECTION_MARKERS if m.lower() in ext_low))

    ents = _sensitive_entities(ep.proposed_action)
    trusted_low = trusted.lower()
    if ents:
        q4 = min(1.0 if e.lower() in trusted_low else 0.0 for e in ents)
    else:
        q4 = 1.0

    act_low = ep.proposed_action.lower()
    q5 = 0.0
    for v in _VERBS:
        if v in ext_low and v in act_low:
            q5 = 1.0
            break

    q6 = float(len(externals))

    # q7: email-like entities introduced by EXTERNAL observations that never
    # appear in TRUSTED ones (roster/audience smuggled in via tool output).
    ext_emails = {t for t in _content_tokens(ext_all) if "@" in t}
    trusted_emails = {t for t in _content_tokens(trusted) if "@" in t}
    q7 = float(len(ext_emails - trusted_emails))

    # q8: sensitive entities used by the action that TRUSTED observations
    # never authorize (exfiltrated recipient, redirected path).
    q8 = float(sum(1 for e in ents if e.lower() not in trusted_low))

    return [q1, q2, q3, q4, q5, q6, q7, q8]


def role_tagged_text(ep):
    trusted = _trusted_text(ep)
    ext = " ".join(_external_texts(ep))
    return ("[TRUSTED] " + trusted + " [EXTERNAL] " + ext +
            " [PROPOSED] " + ep.proposed_action +
            " [ALTERNATIVE] " + ep.alternative_action)


class ProvenanceFeaturizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.tfidf_ = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\[?[A-Za-z@._/\-']+\]?",
            ngram_range=(1, 2), min_df=2)
        self.tfidf_.fit([role_tagged_text(ep) for ep in X])
        return self

    def transform(self, X):
        tf = self.tfidf_.transform([role_tagged_text(ep) for ep in X])
        pf = np.array([provenance_features(ep) for ep in X], dtype=float)
        return hstack([tf, csr_matrix(pf)])


def _label(ep):
    return 1 if ep.label == UNSAFE else 0


def train(episodes):
    model = Pipeline([
        ("feat", ProvenanceFeaturizer()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, C=1.0), method="isotonic", cv=3)),
    ])
    model.fit(episodes, [_label(ep) for ep in episodes])
    return model


def p_unsafe(model, ep):
    return float(model.predict_proba([ep])[0][1])


def decide(model, ep):
    """True = BLOCK the proposed action (commit the safe alternative).

    argmax over {proposed, alternative} with the alternative a known-safe
    no-op of fixed value V_alt = 0.5: commit proposed iff P(safe) > 0.5.
    """
    return p_unsafe(model, ep) >= BLOCK_THRESHOLD
