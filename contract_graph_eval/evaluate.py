"""Run the judges on a prediction and summarise how much they agree."""
import json
from typing import Any, Dict, Iterable, List, Optional

from .judges import Completeness, ExactMatch, GraphEditDistance, Hybrid, ParseSuccess, StructuralF1
from .reference_free import reference_free_scores
from .uncertainty import disagreement, mean_risk

# Judges whose scores are compared to measure disagreement, and to estimate risk.
DISAGREEMENT_POOL = ("structural_f1", "semantic_similarity", "openai_issues", "hybrid", "completeness", "graph_edit_distance")
RISK_POOL = ("structural_f1", "semantic_similarity", "completeness", "graph_edit_distance", "openai_issues")
ERROR_THRESHOLD = 0.5


def default_judges(embedding_model=None, llm_judge=None) -> List[Any]:
    """The judges used in the paper.

    The structural judges need nothing extra. Pass a sentence-transformers model
    to add the semantic-similarity and hybrid judges, and an ``LLMJudge`` to add
    the LLM judge.
    """
    structural = StructuralF1()
    judges = [ExactMatch(), structural, ParseSuccess(), Completeness(), GraphEditDistance()]
    if embedding_model is not None:
        from .semantic import SemanticSimilarity
        semantic = SemanticSimilarity(embedding_model)
        judges += [semantic, Hybrid(structural, semantic, alpha=0.5)]
    if llm_judge is not None:
        judges.append(llm_judge)
    return judges


def _as_graph(prediction: Any) -> Any:
    """Accept a graph, JSON text, or a record such as {"graph": ..., "raw_text": ...}."""
    if isinstance(prediction, dict) and "nodes" not in prediction and ("graph" in prediction or "raw_text" in prediction):
        return prediction["graph"] if prediction.get("graph") is not None else prediction.get("raw_text")
    return prediction


def evaluate(prediction: Any, reference: Any, clause_text: str = "", instruction: str = "",
             judges: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """Score one predicted graph against a reference graph.

    Returns the score of every judge, the reference-free checks, the
    disagreement statistics over the judges, the mean risk, and whether any
    judge scored the prediction below 0.5.
    """
    pred = _as_graph(prediction)
    judges = list(judges) if judges is not None else default_judges()
    scores, details = {}, {}
    for judge in judges:
        if judge.name == "openai_issues":
            result = judge.score(pred, reference, clause_text=clause_text, original_prompt=instruction)
        else:
            result = judge.score(pred, reference)
        details[judge.name] = result
        scores[judge.name] = result.get("score")

    pool = [name for name in DISAGREEMENT_POOL if scores.get(name) is not None]
    pooled = [scores[name] for name in pool]
    uncertainty = disagreement(pooled)
    uncertainty["judges"] = pool
    return {
        "judge_scores": scores,
        "reference_free_judges": reference_free_scores(pred if isinstance(pred, dict) else None),
        "uncertainty": uncertainty,
        "mean_risk": mean_risk(scores[name] for name in RISK_POOL if scores.get(name) is not None),
        "evaluation": {
            "has_error": bool(any(s < ERROR_THRESHOLD for s in pooled)),
            "mean_score": float(sum(pooled) / len(pooled)) if pooled else None,
            "structural_f1": scores.get("structural_f1"),
        },
        "details": details,
    }


def evaluate_records(records: Iterable[Dict[str, Any]], judges: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
    """Score records that each hold ``prediction`` and ``gold_graph`` (or ``reference``).

    ``clause`` ({"text": ...}) and ``instruction`` are passed to the LLM judge when present.
    """
    judges = list(judges) if judges is not None else default_judges()
    out = []
    for record in records:
        clause = record.get("clause")
        result = evaluate(record.get("prediction"), record.get("gold_graph", record.get("reference")),
                          clause_text=clause.get("text", "") if isinstance(clause, dict) else str(clause or ""),
                          instruction=record.get("instruction", ""), judges=judges)
        result.pop("details")
        out.append({**{k: record[k] for k in ("sample_id", "id", "contract_id", "clause_id") if k in record}, **result})
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
