"""evaluation/device_accuracy.py's metrics and gate, on hand-made rows (no model)."""

from __future__ import annotations

import pytest
from device_accuracy import GATE, compare, ece, label_metrics, probabilities, requests

from strands_decider.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer
from training.data.format import Example


def _examples(n=4):
    return [
        Example(
            kind="choice",
            state=f"s{i}",
            instructions="q?",
            options=[["a", ""], ["b", ""]],
            label=i % 2,
        )
        for i in range(n)
    ]


def test_probabilities_are_in_canonical_option_order():
    noul = Example(
        kind="noul", state="", instructions="q?", options=[["false", ""], ["true", ""]], label=1
    )
    assert probabilities(NoulAnswer(noul=0.8), noul) == pytest.approx([0.2, 0.8])
    choice = ChoiceAnswer(choice="b", probabilities={"b": 0.7, "a": 0.3}, confidence=0.4)
    assert probabilities(choice, _examples(1)[0]) == [0.3, 0.7]
    levels = [["0", "low"], ["1", "mid"], ["2", "high"]]
    score_ex = Example(kind="score", state="", instructions="q?", options=levels, label=1)
    score = ScoreAnswer(
        score=1.2, legend={}, probabilities={"2": 0.3, "0": 0.1, "1": 0.6}, confidence=0.5
    )
    assert probabilities(score, score_ex) == [0.1, 0.6, 0.3]


def test_each_example_is_one_question_request():
    [r] = requests(_examples(1))
    assert list(r.questions) == ["q"] and r.state == "s0"


def test_ece_is_zero_when_confidence_is_accuracy_and_one_when_it_is_not():
    assert ece([(0.95, True)] * 19 + [(0.95, False)]) == pytest.approx(0.0, abs=1e-9)
    assert ece([(1.0, False)] * 3) == pytest.approx(1.0)


def test_label_metrics():
    ex = _examples(2)
    m = label_metrics(ex, [[0.9, 0.1], [0.2, 0.8]])
    assert m["accuracy"] == 1.0
    assert m["nll"] == pytest.approx(
        -(__import__("math").log(0.9) + __import__("math").log(0.8)) / 2
    )


def test_rounding_passes_and_a_confident_flip_fails():
    ex = _examples(400)
    ref = [[0.9, 0.1] if e.label == 0 else [0.2, 0.8] for e in ex]
    ref[2] = [0.51, 0.49]
    near = [[p0 + 0.0005, p1 - 0.0005] for p0, p1 in ref]
    near[2] = [0.49, 0.51]  # a near-tie flips: one row in 400 changes accuracy by 0.0025
    r = compare(ex, ref, near)
    assert r["pass"], r
    assert [f["row"] for f in r["changed"]] == [2]
    assert r["changed"][0]["margin"] == pytest.approx(0.02)
    far = list(ref)
    far[0] = [0.1, 0.9]  # a confident answer flips
    assert not compare(ex, ref, far)["pass"]
    drift = [[p0 - 0.01, p1 + 0.01] for p0, p1 in ref]  # no confident flip, past mean |dp|
    r = compare(ex, ref, drift)
    assert r["mean_dp"] > GATE["mean_dp"] and not r["pass"]
