from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from prompt import build_prompt
from retrieve import retrieve_chunks


DEFAULT_MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_1",
    "phi3:mini",
]

DEFAULT_QUESTIONS = [
    "What evidence in the corpus describes common symptoms of COVID-19?",
    "What does the corpus say about influenza vaccine effectiveness?",
    "How is latent tuberculosis different from active tuberculosis?",
    "What treatments are discussed for HIV infection?",
    "What complications are associated with severe COVID-19 cases?",
    "What diagnostic approaches are mentioned for tuberculosis?",
    "What risk factors are associated with influenza severity?",
    "What does the corpus say about antiviral therapy for HIV?",
    "How do the retrieved studies describe prevention strategies for tuberculosis spread?",
    "What limitations or uncertainties are reported in the corpus for infectious disease treatment?",
]

TOPIC_HINTS = {
    "covid": "covid19",
    "sars-cov-2": "covid19",
    "influenza": "influenza",
    "flu": "influenza",
    "tuberculosis": "tuberculosis",
    "tb": "tuberculosis",
    "hiv": "hiv",
}


def infer_topic_filter(question: str) -> str | None:
    q = question.lower()
    for needle, topic in TOPIC_HINTS.items():
        if needle in q:
            return topic
    return None


def load_questions(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_QUESTIONS
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict) and "questions" in payload:
        return [str(x) for x in payload["questions"]]
    raise ValueError("Questions file must be a list or an object with a 'questions' field.")


def query_ollama(model: str, prompt: str, base_url: str, timeout: int = 300) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    response = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def run_experiment(
    questions: List[str],
    models: List[str],
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    top_k: int,
    ollama_base_url: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for question in questions:
        topic_filter = infer_topic_filter(question)
        retrieved = retrieve_chunks(
            query=question,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            top_k=top_k,
            topic_filter=topic_filter,
        )
        prompt = build_prompt(question, retrieved)

        for model in models:
            started = time.time()
            try:
                answer = query_ollama(model=model, prompt=prompt, base_url=ollama_base_url)
                error = ""
            except Exception as exc:  # pragma: no cover
                answer = ""
                error = str(exc)
            latency = round(time.time() - started, 3)

            rows.append(
                {
                    "question": question,
                    "topic_filter": topic_filter or "",
                    "model": model,
                    "latency_seconds": latency,
                    "answer": answer,
                    "error": error,
                    "retrieved_context": json.dumps(retrieved, ensure_ascii=False),
                }
            )
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all questions against all three local LLMs via Ollama.")
    parser.add_argument("--questions", help="Optional JSON file with a list of questions.")
    parser.add_argument("--output", default="results/answers.csv")
    parser.add_argument("--persist-dir", default="data/chroma_db")
    parser.add_argument("--collection", default="pubmed_medical_rag")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    rows = run_experiment(
        questions=questions,
        models=args.models,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        ollama_base_url=args.ollama_base_url,
    )
    write_csv(rows, args.output)
    print(f"Saved {len(rows)} model answers to {args.output}")


if __name__ == "__main__":
    main()
