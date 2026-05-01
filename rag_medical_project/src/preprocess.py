from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(sentences: List[str], chunk_size: int = 4, overlap: int = 1) -> List[str]:
    if not sentences:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(sentences):
        chunk = " ".join(sentences[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def record_to_chunks(record: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    title = clean_text(record.get("title", ""))
    abstract = clean_text(record.get("abstract", ""))
    if not title or not abstract:
        return []

    sentences = split_into_sentences(abstract)
    if len(sentences) <= chunk_size:
        text_chunks = [abstract]
    else:
        text_chunks = chunk_sentences(sentences, chunk_size=chunk_size, overlap=overlap)

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(text_chunks):
        enriched = (
            f"Title: {title}\n"
            f"Topic: {record.get('topic', '')}\n"
            f"Year: {record.get('year', '')}\n"
            f"Journal: {record.get('journal', '')}\n"
            f"Abstract chunk: {text}"
        )
        results.append(
            {
                "id": f"{record.get('pmid', 'unknown')}_{i}",
                "text": enriched,
                "chunk_text": text,
                "metadata": {
                    "pmid": record.get("pmid", ""),
                    "title": title,
                    "topic": record.get("topic", ""),
                    "year": record.get("year", ""),
                    "journal": record.get("journal", ""),
                    "authors": record.get("authors", []),
                    "mesh_terms": record.get("mesh_terms", []),
                    "chunk_id": i,
                    "source": "pubmed",
                },
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean PubMed data and create chunked documents.")
    parser.add_argument("--input", default="data/raw_pubmed.json")
    parser.add_argument("--output", default="data/processed_chunks.json")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=1)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records = payload.get("records", [])

    processed: List[Dict[str, Any]] = []
    for record in records:
        processed.extend(record_to_chunks(record, chunk_size=args.chunk_size, overlap=args.overlap))

    output = {
        "num_input_records": len(records),
        "num_chunks": len(processed),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "chunks": processed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(processed)} chunks to {output_path}")


if __name__ == "__main__":
    main()
