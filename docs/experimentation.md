# Experimentation

This repository includes scripts and utilities to evaluate system variants (graphRAG, agentRAG, naiveRAG) and generate evaluation data.

## 1) Generate ground truth

Use the helper script to create or refresh labeled data used for evaluation:

```bash
uv run python utils/generate_ground_truth.py
```

Inputs and outputs are configured in the script and/or local settings files.

## 2) Compare systems

Run comparative evaluations across systems:

```bash
uv run python scripts/experiments/compare_systems.py
```

Results are written under [results/](results/), including comparison tables and raw outputs.

## 3) RAGAS evaluation (graphRAG / agentRAG / naiveRAG)

Run RAGAS-based evaluation using the generated ground truth:

```bash
uv run python scripts/experiments/ragas_compare.py --systems graph,agent,naive --ground-truth results/ground_truth_answers.json --out results/ragas
```

Outputs:

- `results/ragas/{system}_predictions.jsonl`
- `results/ragas/{system}_ragas_scores.json`

