from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd


def maybe_run_ragas(answers_path: str, output_path: str) -> bool:
    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import faithfulness, answer_relevancy  # type: ignore
        from ragas.llms import LangchainLLMWrapper  # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
        from ragas.run_config import RunConfig  # type: ignore
        from langchain_ollama import ChatOllama, OllamaEmbeddings  # type: ignore
        from datasets import Dataset  # type: ignore
    except Exception as e:
        print(f"RAGAS import failed: {e}")
        return False

    df = pd.read_csv(answers_path)
    contexts: List[List[str]] = []
    for raw in df["retrieved_context"].fillna(""):
        try:
            import json

            parsed = json.loads(raw)
            contexts.append([item.get("document", "") for item in parsed])
        except Exception:
            contexts.append([])

    ragas_df = pd.DataFrame(
        {
            "question": df["question"],
            "answer": df["answer"],
            "contexts": contexts,
        }
    )

    dataset = Dataset.from_pandas(ragas_df)
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model="llama3.1:8b-instruct-q4_K_M",
            temperature=0,
            base_url="http://localhost:11434",
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )
    )

    run_config = RunConfig(
        timeout=600,
        max_retries=2,
        max_wait=30,
        max_workers=1,
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    result.to_pandas().to_csv(output_path, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional RAGAS evaluation.")
    parser.add_argument("--answers", default="results/answers.csv")
    parser.add_argument("--ragas-output", default="results/ragas_scores.csv")
    parser.add_argument("--try-ragas", action="store_true")
    args = parser.parse_args()


    if args.try_ragas:
        ok = maybe_run_ragas(args.answers, args.ragas_output)
        if ok:
            print(f"RAGAS results written: {args.ragas_output}")
        else:
            print("RAGAS not available in this environment; skipped.")


if __name__ == "__main__":
    main()
