#!/usr/bin/env python3
"""Pick the predictions most likely to be wrong, for human review or refinement.

    python examples/selective_review.py                    # review 10% of the benchmark
    python examples/selective_review.py --budget 0.2
    python examples/selective_review.py --score uncertainty.cv
    python examples/selective_review.py --file my_scored_predictions.jsonl

Samples are ranked by a risk score and the top share is selected. The default
score is the calibrated error probability; any numeric field can be used,
written as a dotted path such as ``uncertainty.cv``.

Standard library only.
"""
import argparse
import json
import signal
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"


def read(record, dotted):
    value = record
    for key in dotted.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def main():
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)      # exit quietly when the output is piped to head or less
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="JSON Lines file of scored samples (default: the benchmark)")
    ap.add_argument("--budget", type=float, default=0.10, help="share of samples to review (default 0.10)")
    ap.add_argument("--score", default="error_prediction.p_error_calibrated", help="field to rank by, highest first")
    args = ap.parse_args()

    path = Path(args.file) if args.file else max(BENCHMARK_DIR.glob("*.jsonl"))
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    scored = [(read(r, args.score), r) for r in rows]
    scored = [(s, r) for s, r in scored if isinstance(s, (int, float))]
    scored.sort(key=lambda sr: sr[0], reverse=True)
    k = max(1, round(args.budget * len(scored)))

    print(f"{path.name}: {len(scored)} samples, reviewing {k} ({args.budget:.0%}) ranked by {args.score}\n")
    print(f"{'risk':>6}  {'disagreement':>12}  {'hybrid':>6}  clause")
    for score, r in scored[:k]:
        cv = read(r, "uncertainty.cv")
        hybrid = read(r, "judge_scores.hybrid")
        text = " ".join(str(read(r, "clause.text") or "").split())[:70]
        print(f"{score:>6.3f}  {cv if cv is None else format(cv, '>12.3f')}  {hybrid if hybrid is None else format(hybrid, '>6.3f')}  {r.get('clause_id', '')}: {text}")


if __name__ == "__main__":
    main()
