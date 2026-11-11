"""Multi-judge evaluation of contract graphs.

    from contract_graph_eval import evaluate

    result = evaluate(predicted_graph, reference_graph)
    result["judge_scores"]            # one score per judge, in [0, 1]
    result["uncertainty"]["cv"]       # how much the judges disagree
    result["evaluation"]["has_error"]

Reference: M. Dechtiar, D. M. Katz, S. Jaume, H. Wang, "LLM as a Judge for
Evaluating Contract Graphs: Multi-Judge Benchmarking and Agentic
Uncertainty-Aware Refinement," Procedia Computer Science 275 (2026) 513-522.
"""
from .evaluate import (DISAGREEMENT_POOL, ERROR_THRESHOLD, RISK_POOL, default_judges, evaluate,
                       evaluate_records, read_jsonl)
from .judges import Completeness, ExactMatch, GraphEditDistance, Hybrid, ParseSuccess, StructuralF1
from .reference_free import reference_free_scores
from .uncertainty import calibrate, disagreement, mean_risk, select_for_review

__version__ = "0.1.0"

__all__ = [
    "evaluate", "evaluate_records", "default_judges", "read_jsonl",
    "ExactMatch", "StructuralF1", "ParseSuccess", "Completeness", "GraphEditDistance", "Hybrid",
    "reference_free_scores", "disagreement", "mean_risk", "select_for_review", "calibrate",
    "DISAGREEMENT_POOL", "RISK_POOL", "ERROR_THRESHOLD",
]
