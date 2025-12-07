# ContractGraphEval: Multi-Judge Benchmarking and Agentic Uncertainty-Aware Refinement

[![Paper](https://img.shields.io/badge/Paper-ACLing%202025-blue)](https://doi.org/10.1016/j.procs.2025.xx.xxx)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official repository for **"LLM as a Judge for Evaluating Contract Graphs: Multi-Judge Benchmarking and Agentic Uncertainty-Aware Refinement"** (ACLing 2025).

## Overview

This repository provides:

- **ContractGraphEval v1.0**: The first benchmark for clause-level legal knowledge graph extraction with calibrated uncertainty scores
- **Multi-Judge Evaluation Pipeline**: 17 complementary judges (9 reference-based + 8 reference-free)
- **Calibration Analysis**: ECE, ROC-AUC, and bootstrap confidence intervals
- **Agentic Refinement**: Uncertainty-driven Planner–Refiner–Verifier loop

### Key Results

| Metric | Value |
|--------|-------|
| CV-based Error Detection AUC | **0.982** |
| Calibration (ECE) | 0.027 |
| Error Recall @ 10% Budget | 91% |
| Cost Reduction vs Blind Refinement | **90%** |

## Repository Structure

```
ContractGraphEval/
├── README.md
├── requirements.txt
├── LICENSE
├── benchmark/
│   └── ContractGraphEval_v1.0.jsonl      # 150 annotated samples
├── notebooks/
│   ├── 1_multiJudgeEvalOnPredictions.ipynb   # Multi-judge evaluation pipeline
│   ├── 2_calibrationEval.ipynb               # Calibration & ROC analysis
│   └── 3_uncertaintyAwareAgenticRefinement.ipynb  # Selective refinement
└── paper/
    └── Multi-Judge-Benchmarking-Agentic-Uncertainty-Aware-Refinement.pdf
```

## Benchmark: ContractGraphEval v1.0

Each sample in `ContractGraphEval_v1.0.jsonl` contains:

| Field | Description |
|-------|-------------|
| `sample_id` | Unique identifier |
| `clause` | Source contract clause text |
| `gold_graph` | Expert-annotated knowledge graph |
| `prediction` | Model-generated graph (GRPO-trained Llama-3.1-8B) |
| `judge_scores` | 9 reference-based judge scores |
| `reference_free_judges` | 8 reference-free judge scores |
| `uncertainty` | CV, entropy, variance, and 5 other disagreement metrics |
| `evaluation` | `has_error` label and aggregate scores |

### Judge Categories

**Reference-Based (Research Evaluation)**
- Structural F1, Graph Edit Distance, Exact Match
- Semantic Similarity, Completeness, Hybrid Score
- Parse Success, OpenAI Issues, Gemini Changes

**Reference-Free (Production Deployment)**
- Unused Definitions, Undefined References, Duplicate Definitions
- Orphan Nodes, Isolated Components
- Domain-Range Validity, Ontology Compliance

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ContractGraphEval.git
cd ContractGraphEval
pip install -r requirements.txt
```

### Load the Benchmark

```python
import json

samples = []
with open('benchmark/ContractGraphEval_v1.0.jsonl', 'r') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# Access uncertainty metrics
for sample in samples[:3]:
    cv = sample['uncertainty']['cv']
    has_error = sample['evaluation']['has_error']
    print(f"CV: {cv:.3f}, Error: {has_error}")
```

### Compute Selective Refinement Threshold

```python
import numpy as np

cvs = [s['uncertainty']['cv'] for s in samples]
errors = [s['evaluation']['has_error'] for s in samples]

# CV > 0.15 threshold captures 91% of errors at 10% budget
threshold = 0.15
flagged = [cv > threshold for cv in cvs]
recall = sum(e and f for e, f in zip(errors, flagged)) / sum(errors)
print(f"Error recall at CV > {threshold}: {recall:.1%}")
```

## Notebooks

### 1. Multi-Judge Evaluation (`1_multiJudgeEvalOnPredictions.ipynb`)

Runs all 17 judges on model predictions:
- Structural judges (F1, GED, Parse Success)
- Semantic judges (embedding similarity, completeness)
- LLM judges (OpenAI, Gemini issue detection)
- Contract linters (orphans, broken references)

**Configuration:**
```python
# Set your data paths
PREDICTIONS_DIR = Path("./data/predictions")
OUTPUT_DIR = Path("./outputs/judge_scores")
```

### 2. Calibration Analysis (`2_calibrationEval.ipynb`)

Statistical validation of uncertainty signals:
- ROC-AUC for each disagreement metric
- Expected Calibration Error (ECE)
- Bootstrap confidence intervals (2,000 resamples)
- Leave-One-Judge-Out (LOJO) analysis

**Key Output:** Table 1 and Figure 1 from the paper

### 3. Agentic Refinement (`3_uncertaintyAwareAgenticRefinement.ipynb`)

Implements the Planner–Refiner–Verifier loop:
- Routes predictions based on CV threshold
- Applies GPT-4 refinement only to high-uncertainty samples
- Evaluates quality improvement and cost savings

**Configuration:**
```python
# Set your data paths
PREDICTIONS_PATH = "./data/predictions.jsonl"
JUDGE_SCORES_PATH = "./outputs/judge_scores_summary.csv"
```

## Configuration

The notebooks use Google Colab by default. To run locally:

1. Update the path variables in each notebook's Configuration section
2. Set API keys as environment variables:
   ```bash
   export OPENAI_API_KEY="your-key"
   export GEMINI_API_KEY="your-key"  # optional
   ```

## Citation

If you use ContractGraphEval or our multi-judge framework, please cite:

```bibtex
@inproceedings{dechtiar2025llmjudge,
  title={LLM as a Judge for Evaluating Contract Graphs: Multi-Judge Benchmarking and Agentic Uncertainty-Aware Refinement},
  author={Dechtiar, Moriya and Katz, Daniel Martin and Jaume, Sylvain and Wang, Hongming},
  booktitle={Proceedings of the 7th International Conference on AI in Computational Linguistics (ACLing 2025)},
  year={2025},
  publisher={Elsevier},
  note={Procedia Computer Science}
}
```

## Related Work

This benchmark builds on [GRAPH-GRPO-LEX](https://ssrn.com/abstract=5566538), which introduced the contract graph ontology and GRPO training methodology.

## License

This work is licensed under [CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/).

- **Benchmark data**: Available for research use
- **Code**: MIT License (see notebooks)

## Contact

- **Moriya Dechtiar** - moriya.dechtiar@alumni.harvard.edu
- Issues and PRs welcome!

## Acknowledgments

We thank the Harvard NLP group, Illinois Tech Chicago-Kent College of Law, MIT, and Stanford CodeX for their support.
