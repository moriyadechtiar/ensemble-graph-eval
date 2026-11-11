"""Tests for contract_graph_eval. Run with:  python -m pytest -q"""
import json
import types

import numpy as np
import pytest

from contract_graph_eval import (Completeness, ExactMatch, GraphEditDistance, Hybrid, ParseSuccess, StructuralF1,
                                 default_judges, disagreement, evaluate, evaluate_records, mean_risk,
                                 reference_free_scores, select_for_review)
from contract_graph_eval.__main__ import main
from contract_graph_eval.llm import LLMJudge
from contract_graph_eval.semantic import SemanticSimilarity

GOLD = {"contract_id": "C", "nodes": [
    {"id": "4", "node_type": "CLAUSE", "title": "Permitted Disclosure", "level": 1},
    {"id": "4.1", "node_type": "CLAUSE", "title": "4.1", "level": 2},
    {"id": "term:Affiliate", "node_type": "DEFINED_TERM", "name": "Affiliate"},
    {"id": "party:Recipient", "node_type": "PARTY", "name": "Recipient"}],
    "edges": [{"src": "4.1", "tgt": "4", "type": "IS_PART_OF"},
              {"src": "4.1", "tgt": "term:Affiliate", "type": "USES"},
              {"src": "4.1", "tgt": "party:Recipient", "type": "MENTIONS_PARTY"}]}


def without(graph, node_id):
    return {"contract_id": graph["contract_id"], "nodes": [n for n in graph["nodes"] if n["id"] != node_id],
            "edges": [e for e in graph["edges"] if node_id not in (e["src"], e["tgt"])]}


@pytest.mark.parametrize("judge", [ExactMatch(), StructuralF1(), ParseSuccess(), Completeness(), GraphEditDistance()])
def test_identical_graph_scores_one(judge):
    assert judge.score(GOLD, GOLD)["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("judge", [StructuralF1(), Completeness(), GraphEditDistance()])
def test_missing_node_lowers_the_score(judge):
    score = judge.score(without(GOLD, "party:Recipient"), GOLD)["score"]
    assert 0.0 < score < 1.0


def test_judges_accept_json_text():
    assert StructuralF1().score(json.dumps(GOLD), json.dumps(GOLD))["score"] == pytest.approx(1.0)


def test_unparseable_prediction_scores_zero():
    for judge in (ExactMatch(), StructuralF1(), ParseSuccess(), Completeness(), GraphEditDistance()):
        assert judge.score("not a graph", GOLD)["score"] == 0.0


def test_parse_success_weights():
    assert ParseSuccess().score({"contract_id": "C", "nodes": GOLD["nodes"], "edges": []}, GOLD)["score"] == pytest.approx(0.6)


class FakeEmbedder:
    """Bag-of-characters vectors: identical texts get similarity 1."""
    def encode(self, texts):
        return np.array([[t.count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in (s.lower() for s in texts)], dtype=float)


def test_semantic_and_hybrid_judges():
    semantic = SemanticSimilarity(FakeEmbedder())
    assert semantic.score(GOLD, GOLD)["score"] == pytest.approx(1.0)
    hybrid = Hybrid(StructuralF1(), semantic).score(without(GOLD, "term:Affiliate"), GOLD)
    assert hybrid["score"] == pytest.approx(0.5 * hybrid["structural_score"] + 0.5 * hybrid["semantic_score"])


def test_llm_judge_parses_the_reply():
    reply = json.dumps({"score": "0.4", "issues_found": 2, "issue_list": ["missing node", "wrong edge"], "explanation": "two issues"})
    seen = {}
    def create(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=reply))])
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    result = LLMJudge(client=client).score(GOLD, GOLD, clause_text="4.1 The Recipient may disclose to its Affiliates.")
    assert result["score"] == pytest.approx(0.4) and result["issues_found"] == 2
    assert "CLAUSE_TEXT:\n4.1 The Recipient" in seen["messages"][1]["content"]


def test_disagreement_statistics():
    agree, split = disagreement([0.9, 0.9, 0.9, 0.9]), disagreement([1.0, 0.1, 0.9, 0.2])
    assert agree["cv"] == pytest.approx(0.0, abs=1e-6) and agree["range"] == 0.0
    assert split["cv"] > 0.5 and split["range"] == pytest.approx(0.9) and split["n_judges"] == 4
    assert disagreement([None, 0.5])["n_judges"] == 1


def test_mean_risk_and_selection():
    assert mean_risk([1.0, 0.5]) == pytest.approx(0.25)
    assert select_for_review([0.1, 0.9, 0.3, 0.8], budget=0.5) == [1, 3]


def test_reference_free_scores():
    assert set(reference_free_scores(GOLD).values()) <= {0.0, 1.0} and reference_free_scores(GOLD)["domain_range_validity"] == 1.0
    wrong = json.loads(json.dumps(GOLD)); wrong["edges"].append({"src": "4.1", "tgt": "party:Recipient", "type": "USES"})
    assert reference_free_scores(wrong)["domain_range_validity"] == pytest.approx(0.75)
    assert set(reference_free_scores(None).values()) == {0.0}


def test_evaluate_flags_a_poor_prediction():
    good, poor = evaluate(GOLD, GOLD), evaluate({"contract_id": "C", "nodes": [GOLD["nodes"][0]], "edges": []}, GOLD)
    assert good["evaluation"]["has_error"] is False and poor["evaluation"]["has_error"] is True
    assert poor["mean_risk"] > good["mean_risk"] and poor["uncertainty"]["n_judges"] == 3


def test_evaluate_with_all_judges():
    judges = default_judges(embedding_model=FakeEmbedder())
    result = evaluate({"graph": GOLD, "raw_text": json.dumps(GOLD)}, GOLD, judges=judges)
    assert set(result["judge_scores"]) == {"exact_match", "structural_f1", "parse_success", "completeness",
                                           "graph_edit_distance", "semantic_similarity", "hybrid"}
    assert result["uncertainty"]["judges"] == ["structural_f1", "semantic_similarity", "hybrid", "completeness", "graph_edit_distance"]


def test_command_line(tmp_path, capsys):
    src, out = tmp_path / "preds.jsonl", tmp_path / "scored.jsonl"
    records = [{"sample_id": "a", "prediction": GOLD, "gold_graph": GOLD},
               {"sample_id": "b", "prediction": {"graph": without(GOLD, "4"), "raw_text": ""}, "gold_graph": GOLD}]
    src.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert main([str(src), "--out", str(out), "--budget", "0.5"]) == 0
    assert "2 predictions scored" in capsys.readouterr().out
    scored = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["sample_id"] for r in scored] == ["a", "b"] and scored[0]["mean_risk"] < scored[1]["mean_risk"]
    assert evaluate_records(records)[0]["judge_scores"]["structural_f1"] == pytest.approx(1.0)
