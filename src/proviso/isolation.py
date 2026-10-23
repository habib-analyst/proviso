"""Isolation judge baseline: judges the proposed action in isolation.

Sees ONLY the proposed action text. No provenance, no observations.
TF-IDF + logistic regression.

Decision rule: BLOCK iff P(unsafe | action text) >= 0.5, else commit the
proposed action. This is argmax over {proposed, alternative} with the
alternative a known-safe no-op whose value is fixed by design at
V_alt = 0.5 (safe but unhelpful): commit the proposed action iff it is
more likely safe than not.

Calibration note (measured 2026-10-22): isotonic calibration on this
text-duplicated data collapses (one fitted calibrator mapped every raw
score to the constant 0.3503, the global base rate) and Platt scaling
distorts the duplicated-text group (raw 0.41 at 42% base rate -> 0.59).
Both were tried and rejected; the judge uses raw logistic scores, which
track the group base rates honestly. The controller keeps isotonic
calibration (verified sane: 4 thresholds spanning 0.0 -> 1.0).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .traces import UNSAFE

BLOCK_THRESHOLD = 0.5  # == V_alt, the fixed value of the safe no-op


def _label(ep):
    return 1 if ep.label == UNSAFE else 0


def train(episodes):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True,
                                  token_pattern=r"(?u)[a-zA-Z@._/\-']+",
                                  ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    model.fit([ep.proposed_action for ep in episodes],
              [_label(ep) for ep in episodes])
    return model


def p_unsafe(model, action_text):
    return float(model.predict_proba([action_text])[0][1])


def decide(model, ep):
    """True = BLOCK the proposed action (commit the safe alternative)."""
    return p_unsafe(model, ep.proposed_action) >= BLOCK_THRESHOLD
