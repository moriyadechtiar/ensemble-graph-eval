# ContractGraphEval: Multi-Judge Benchmarking and Agentic Uncertainty-Aware Refinement

[![Paper](https://img.shields.io/badge/Paper-Procedia%20CS%20275%20(2026)-blue)](https://doi.org/10.1016/j.procs.2026.01.060)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/moriyad/ContractGraphEval)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official repository for **"LLM as a Judge for Evaluating Contract Graphs: Multi-Judge Benchmarking and Agentic Uncertainty-Aware Refinement"**, presented at ACLing 2025 and published in *Procedia Computer Science* 275 (2026) 513–522.

## Overview

- **ContractGraphEval**: 150 contract clauses, each with a reference graph, a model-predicted graph, judge scores and uncertainty estimates. Also on [Hugging Face](https://huggingface.co/datasets/moriyad/ContractGraphEval).
- **Multi-judge evaluation**: the `contract_graph_eval` package scores a predicted graph with eight reference-based judges and seven reference-free checks, and measures how much the judges disagree.
- **Calibration analysis**: ROC-AUC, expected calibration error, and bootstrap confidence intervals.
- **Agentic refinement**: an uncertainty-driven Planner–Refiner–Verifier loop that refines only the samples most likely to be wrong.

## Quick start

```bash
git clone https://github.com/moriyadechtiar/ensemble-graph-eval.git
cd ensemble-graph-eval
python examples/selective_review.py --budget 0.10      # standard library only
```

This ranks the benchmark samples by estimated risk and lists the share you would send for review. Point it at your own scored predictions with `--file`, or rank by another field with `--score uncertainty.cv`.

Load the benchmark:

```python
from datasets import load_dataset

ds = load_dataset("moriyad/ContractGraphEval", split="validation")
s = ds[0]
s["clause"]["text"], s["judge_scores"]["hybrid"], s["error_prediction"]["p_error_calibrated"]
```

The same data is in `benchmark/` as JSON Lines. Field descriptions are on the dataset card. Results are reported in the [paper](https://doi.org/10.1016/j.procs.2026.01.060).

## Evaluate your own predictions

```bash
pip install -e .                                   # needs only numpy
python -m contract_graph_eval examples/example_predictions.jsonl
python -m contract_graph_eval your_predictions.jsonl --out scored.jsonl
```

Each line of the input is a JSON object with a `prediction` and a `gold_graph`, as in the example file. The command scores every prediction with each judge, measures how much the judges disagree, and lists the highest-risk predictions for review.

From Python:

```python
from contract_graph_eval import evaluate

result = evaluate(predicted_graph, reference_graph)
result["judge_scores"]             # one score per judge, in [0, 1]
result["uncertainty"]["cv"]        # disagreement between the judges
result["mean_risk"]                # used to rank predictions for review
result["reference_free_judges"]    # checks that need no reference graph
```

| Judge | Measures | Needs |
|---|---|---|
| `structural_f1` | Node and edge overlap with the reference | — |
| `completeness` | How much of the reference the prediction covers | — |
| `graph_edit_distance` | Edits needed to turn the prediction into the reference | — |
| `exact_match`, `parse_success` | Identical graph; valid, well-formed output | — |
| `semantic_similarity`, `hybrid` | Embedding similarity, and its average with `structural_f1` | `pip install -e ".[semantic]"`, then `--semantic` |
| `openai_issues` | An LLM lists the issues and gives a score | `pip install -e ".[llm]"`, `OPENAI_API_KEY`, then `--llm` |

`contract_graph_eval.calibrate` turns risk into a probability of error with out-of-sample isotonic calibration (`pip install -e ".[calibration]"`). Tests: `python -m pytest -q`.

## Repository structure

```
ensemble-graph-eval/
├── contract_graph_eval/   evaluation package: judges, uncertainty, review selection, command line
├── tests/                 unit tests
├── benchmark/             ContractGraphEval (JSON Lines)
├── notebooks/             evaluation, calibration, and refinement notebooks
├── examples/              example predictions and selective_review.py
└── paper/                 the published paper
```

## Notebooks

- **`multiJudgeEvalOnPredictions.ipynb`**: runs the judges on model predictions.
- **`calibrationEval.ipynb`**: ROC-AUC, calibration error, bootstrap intervals, and the review-budget table.
- **`uncertaintyAwareAgenticRefinement.ipynb`**: the Planner–Refiner–Verifier loop.

The notebooks were written for Google Colab and read from Google Drive paths. To run them elsewhere, update the path variables in each Configuration cell, install `requirements.txt`, and set `OPENAI_API_KEY`.

## Citation

```bibtex
@article{dechtiar2026llmjudge,
  title   = {LLM as a Judge for Evaluating Contract Graphs: Multi-Judge Benchmarking and Agentic Uncertainty-Aware Refinement},
  author  = {Dechtiar, Moriya and Katz, Daniel Martin and Jaume, Sylvain and Wang, Hongming},
  journal = {Procedia Computer Science},
  volume  = {275},
  pages   = {513--522},
  year    = {2026},
  doi     = {10.1016/j.procs.2026.01.060}
}
```

## Related repositories

| Repository | Paper | Contents |
|---|---|---|
| [ContractSmellsAutoLabelDetection](https://github.com/moriyadechtiar/ContractSmellsAutoLabelDetection) | *Machine Learning with Applications* 20 (2025) 100639 | Contract-smell labeling and detection |
| [graph-grpo-lex](https://github.com/moriyadechtiar/graph-grpo-lex) | IEEE ICDMW 2025, pp. 767–776 | Contract graphs, gated GRPO training, contract linter |
| [ensemble-graph-eval](https://github.com/moriyadechtiar/ensemble-graph-eval) | *Procedia Computer Science* 275 (2026) 513–522 | ContractGraphEval benchmark and multi-judge evaluation |

## License

- **Code** (notebooks and scripts): MIT. See [`LICENSE`](LICENSE).
- **Benchmark data and paper**: CC BY-NC-ND 4.0. See [`LICENSE-DATA.md`](LICENSE-DATA.md).
- Clause text comes from [CUAD](https://www.atticusprojectai.org/cuad) (The Atticus Project, CC BY 4.0).

## Contact

Moriya Dechtiar, moriya.dechtiar@alumni.harvard.edu. Issues and pull requests are welcome.

## Acknowledgments

We thank the Harvard NLP group, Illinois Tech Chicago-Kent College of Law, MIT, and Stanford CodeX for their support.
