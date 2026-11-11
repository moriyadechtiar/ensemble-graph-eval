"""Command line:  python -m contract_graph_eval PREDICTIONS.jsonl [options]

Each line of the input is a JSON object with ``prediction`` and ``gold_graph``.
"""
import argparse
import json
import signal
import sys

from .evaluate import default_judges, evaluate_records, read_jsonl
from .uncertainty import select_for_review


def main(argv=None) -> int:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(prog="contract_graph_eval", description="Score predicted contract graphs with several judges.")
    ap.add_argument("predictions", help="JSON Lines file with 'prediction' and 'gold_graph' in each record")
    ap.add_argument("--out", help="write the scored records to this JSON Lines file")
    ap.add_argument("--semantic", action="store_true", help="add the semantic-similarity and hybrid judges (needs sentence-transformers)")
    ap.add_argument("--llm", action="store_true", help="add the LLM judge (needs openai and OPENAI_API_KEY)")
    ap.add_argument("--budget", type=float, default=0.10, help="share of predictions to list for review (default 0.10)")
    args = ap.parse_args(argv)

    embedding_model = llm_judge = None
    if args.semantic:
        from .semantic import load_embedding_model
        embedding_model = load_embedding_model()
    if args.llm:
        from .llm import LLMJudge
        llm_judge = LLMJudge()

    records = read_jsonl(args.predictions)
    scored = evaluate_records(records, default_judges(embedding_model, llm_judge))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in scored:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    names = [n for n in scored[0]["judge_scores"] if scored[0]["judge_scores"][n] is not None] if scored else []
    print(f"{len(scored)} predictions scored")
    for name in names:
        values = [r["judge_scores"][name] for r in scored if r["judge_scores"].get(name) is not None]
        print(f"  {name:<22}{sum(values) / len(values):.3f}")
    flagged = sum(r["evaluation"]["has_error"] for r in scored)
    print(f"  flagged (a judge below 0.5): {flagged}")
    selected = select_for_review([r["mean_risk"] for r in scored], args.budget)
    print(f"\nHighest-risk {len(selected)} for review:")
    for i in selected:
        r = scored[i]
        ident = r.get("sample_id") or r.get("id") or r.get("clause_id") or i
        print(f"  risk {r['mean_risk']:.3f}  disagreement {r['uncertainty']['cv']:.3f}  {str(ident)[-70:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
