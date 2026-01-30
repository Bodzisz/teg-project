#!/usr/bin/env python3
"""
Run RAGAS evaluation for graphRAG / agentRAG / naiveRAG.
Creates per-system predictions (JSONL) and writes RAGAS scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from utils.experiment_logging import collect_prompt_sources, write_experiment_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("ground_truth_answers", [])


def normalize_contexts(context: Any) -> List[str]:
    if not context:
        return []
    if isinstance(context, list):
        return [json.dumps(c, ensure_ascii=False) if not isinstance(c, str) else c for c in context]
    if isinstance(context, dict):
        return [json.dumps(context, ensure_ascii=False)]
    return [str(context)]


from scripts.experiments.system_init import (
    init_agentrag,
    init_graphrag,
    init_naiverag,
    run_agent_query,
    run_graph_query,
    run_naive_query,
)


def ensure_ragas_imports():
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            AnswerRelevancy,
            FactualCorrectness,
        )
        try:
            from ragas.metrics import AnswerCorrectness
        except Exception:
            AnswerCorrectness = None

        from ragas.llms import LangchainLLMWrapper
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except Exception:
            LangchainEmbeddingsWrapper = None

        return {
            "evaluate": evaluate,
            "EvaluationDataset": EvaluationDataset,
            "SingleTurnSample": SingleTurnSample,
            "metrics": {
                "ContextPrecision": ContextPrecision,
                "ContextRecall": ContextRecall,
                "Faithfulness": Faithfulness,
                "AnswerRelevancy": AnswerRelevancy,
                "FactualCorrectness": FactualCorrectness,
                "AnswerCorrectness": AnswerCorrectness,
            },
            "LangchainLLMWrapper": LangchainLLMWrapper,
            "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        }
    except Exception as e:
        raise RuntimeError(
            "RAGAS or its dependencies are not installed. "
            "Install with: uv run python -m pip install ragas datasets"
        ) from e


def build_samples(questions: List[Dict[str, Any]], runner, system_name: str, SingleTurnSample):
    jsonl_rows = []
    ragas_samples = []
    for i, item in enumerate(questions):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth_answer", "")

        start = time.perf_counter()
        res = runner(question)
        elapsed = time.perf_counter() - start

        jsonl_rows.append({
            "id": f"{system_name}_{i+1}",
            "question": question,
            "answer": res.get("answer", ""),
            "contexts": normalize_contexts(res.get("contexts", [])),
            "ground_truth": ground_truth,
            "metadata": {
                "category": item.get("category"),
                "elapsed_s": elapsed
            }
        })

        ragas_samples.append(
            SingleTurnSample(
                user_input=question,
                response=res.get("answer", ""),
                retrieved_contexts=normalize_contexts(res.get("contexts", [])),
                reference=ground_truth,
            )
        )

    return jsonl_rows, ragas_samples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_evaluator():
    imports = ensure_ragas_imports()

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_EVAL_MODEL", "gpt-4.1"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    llm_wrapper = imports["LangchainLLMWrapper"](llm)
    if imports["LangchainEmbeddingsWrapper"]:
        embeddings_wrapper = imports["LangchainEmbeddingsWrapper"](embeddings)
    else:
        embeddings_wrapper = embeddings

    metrics = [
        imports["metrics"]["ContextPrecision"](llm=llm_wrapper),
        imports["metrics"]["ContextRecall"](llm=llm_wrapper),
        imports["metrics"]["Faithfulness"](llm=llm_wrapper),
        imports["metrics"]["AnswerRelevancy"](llm=llm_wrapper),
        imports["metrics"]["FactualCorrectness"](llm=llm_wrapper),
    ]
    if imports["metrics"]["AnswerCorrectness"]:
        metrics.append(imports["metrics"]["AnswerCorrectness"](llm=llm_wrapper))

    return imports, llm_wrapper, embeddings_wrapper, metrics


def evaluate_with_ragas(ragas_samples, out_path: Path):
    imports, llm_wrapper, embeddings_wrapper, metrics = build_evaluator()

    dataset = imports["EvaluationDataset"](samples=ragas_samples)
    result = imports["evaluate"](
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embeddings_wrapper
    )

    # Convert the result to a JSON-serializable structure if needed (e.g., RAGAS Result -> pandas DataFrame -> dict)
    serializable_result = result
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        try:
            serializable_result = df.to_dict(orient="records")
        except TypeError:
            # Fallback to default orientation if orient="records" is not supported
            serializable_result = df.to_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)

    return result


def summarize_result(result) -> Dict[str, float]:
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c not in {"question", "answer", "contexts", "ground_truth"}]
        return {col: float(df[col].mean()) for col in metric_cols if col in df}
    if isinstance(result, dict):
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    return {}


def write_comparison_table(summaries: Dict[str, Dict[str, float]], out_path: Path):
    if not summaries:
        return

    # union of metrics across systems
    metrics = sorted({m for s in summaries.values() for m in s.keys()})

    lines = ["# RAGAS Comparison", "", "| System | " + " | ".join(metrics) + " |", "|---|" + "|".join(["---"] * len(metrics)) + "|"]

    for system, scores in summaries.items():
        row = [system] + [f"{scores.get(m, 0):.4f}" for m in metrics]
        lines.append("| " + " | ".join(row) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for RAG systems")
    parser.add_argument("--systems", default="graph,agent,naive", help="Comma-separated: graph,agent,naive")
    parser.add_argument("--ground-truth", default="results/ground_truth_answers.json", help="Ground truth JSON file")
    parser.add_argument("--out", default="results/ragas", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions")
    args = parser.parse_args()

    questions = load_ground_truth(Path(args.ground_truth))
    if args.limit and args.limit > 0:
        questions = questions[:args.limit]

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    out_dir = Path(args.out)

    imports = ensure_ragas_imports()
    SingleTurnSample = imports["SingleTurnSample"]

    graph_rag = init_graphrag() if "graph" in systems or "agent" in systems else None
    naive_rag = init_naiverag(querier=True) if "naive" in systems or "agent" in systems else None
    agent_rag = init_agentrag(graph_rag, naive_rag) if "agent" in systems else None

    prompt_sources = collect_prompt_sources({
        "graph.cypher_generation": "src/rag/graph/prompts/cypher_generation.txt",
        "graph.cypher_qa": "src/rag/graph/prompts/cypher_qa.txt",
        "agent.coordinator": "src/rag/agent/prompts/coordinator.txt",
        "agent.planner": "src/rag/agent/prompts/planner.txt",
        "agent.cleanup": "src/rag/agent/prompts/cleanup.txt",
        "naive.system": "src/rag/naive/prompts/naive_rag_system.txt",
    })

    write_experiment_metadata(
        run_name="ragas_compare",
        metadata={
            "systems": systems,
            "ground_truth": str(args.ground_truth),
            "out_dir": str(out_dir),
            "limit": args.limit,
            "models": {
                "openai_model": os.getenv("OPENAI_MODEL"),
                "openai_eval_model": os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
                "openai_embeddings_model": os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),
            },
            "metrics": [
                "ContextPrecision",
                "ContextRecall",
                "Faithfulness",
                "AnswerRelevancy",
                "FactualCorrectness",
                "AnswerCorrectness",
            ],
            "prompts": prompt_sources,
        }
    )

    summaries = {}

    for system in systems:
        if system == "graph":
            jsonl_rows, ragas_samples = build_samples(
                questions, lambda q: run_graph_query(graph_rag, q), "graph", SingleTurnSample
            )
        elif system == "naive":
            jsonl_rows, ragas_samples = build_samples(
                questions, lambda q: run_naive_query(naive_rag, q), "naive", SingleTurnSample
            )
        elif system == "agent":
            jsonl_rows, ragas_samples = build_samples(
                questions, lambda q: run_agent_query(agent_rag, q), "agent", SingleTurnSample
            )
        else:
            raise ValueError(f"Unknown system: {system}")

        predictions_file = out_dir / f"{system}_predictions.jsonl"
        scores_file = out_dir / f"{system}_ragas_scores.json"

        write_jsonl(predictions_file, jsonl_rows)
        result = evaluate_with_ragas(ragas_samples, scores_file)
        summaries[system] = summarize_result(result)

        print(f"✓ {system} predictions: {predictions_file}")
        print(f"✓ {system} RAGAS scores: {scores_file}")

    comparison_table = out_dir / "comparison_table.md"
    write_comparison_table(summaries, comparison_table)
    print(f"✓ RAGAS comparison table: {comparison_table}")


if __name__ == "__main__":
    main()
