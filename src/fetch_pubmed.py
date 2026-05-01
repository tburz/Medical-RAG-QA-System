from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from Bio import Entrez
from Bio import Medline


DEFAULT_TOPICS: Dict[str, str] = {
    "covid19": '(COVID-19 OR SARS-CoV-2) AND (treatment OR symptoms OR diagnosis OR prevention)',
    "influenza": '(influenza OR flu) AND (vaccine OR symptoms OR severity OR prevention)',
    "tuberculosis": '(tuberculosis OR TB) AND (diagnosis OR latent OR active OR treatment)',
    "hiv": '(HIV OR human immunodeficiency virus) AND (therapy OR antiretroviral OR prevention OR diagnosis)',
}


@dataclass
class PubMedRecord:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    topic: str
    query: str
    authors: List[str]
    mesh_terms: List[str]


def search_pmids(query: str, retmax: int) -> List[str]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort="relevance")
    result = Entrez.read(handle)
    handle.close()
    return result.get("IdList", [])


def fetch_medline_records(pmids: List[str]) -> List[dict]:
    if not pmids:
        return []
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()
    return records


def normalize_medline_record(record: dict, topic: str, query: str) -> PubMedRecord | None:
    abstract = " ".join(record.get("AB", "").split())
    title = " ".join(record.get("TI", "").split())
    if not abstract or not title:
        return None

    authors = record.get("AU", []) if isinstance(record.get("AU", []), list) else [record.get("AU", "")]
    mesh_terms = record.get("MH", []) if isinstance(record.get("MH", []), list) else [record.get("MH", "")]
    dp = str(record.get("DP", "")).strip()
    year = dp[:4] if dp else ""

    return PubMedRecord(
        pmid=str(record.get("PMID", "")).strip(),
        title=title,
        abstract=abstract,
        journal=str(record.get("JT", "")).strip(),
        year=year,
        topic=topic,
        query=query,
        authors=[a for a in authors if a],
        mesh_terms=[m for m in mesh_terms if m],
    )


def collect_corpus(email: str, per_topic: int, topics: Dict[str, str], api_key: str | None = None, delay: float = 0.34) -> Dict[str, object]:
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    corpus: List[PubMedRecord] = []
    seen_pmids = set()

    for topic, query in topics.items():
        pmids = search_pmids(query, per_topic * 3)
        time.sleep(delay)
        records = fetch_medline_records(pmids)
        time.sleep(delay)

        kept = 0
        for record in records:
            item = normalize_medline_record(record, topic, query)
            if item is None:
                continue
            if item.pmid in seen_pmids:
                continue
            seen_pmids.add(item.pmid)
            corpus.append(item)
            kept += 1
            if kept >= per_topic:
                break

    return {
        "email": email,
        "per_topic": per_topic,
        "topics": topics,
        "num_records": len(corpus),
        "records": [asdict(r) for r in corpus],
    }


def parse_topic_overrides(topic_args: List[str] | None) -> Dict[str, str]:
    if not topic_args:
        return DEFAULT_TOPICS
    parsed: Dict[str, str] = {}
    for item in topic_args:
        if "=" not in item:
            raise ValueError(f"Invalid --topic override '{item}'. Use topic=query")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for the RAG medical corpus.")
    parser.add_argument(
        "--email",
        default=os.getenv("NCBI_EMAIL"),
        help="Email required by NCBI Entrez. Defaults to the NCBI_EMAIL environment variable.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("NCBI_API_KEY"),
        help="Optional NCBI API key. Defaults to the NCBI_API_KEY environment variable.",
    )
    parser.add_argument("--per-topic", type=int, default=25, help="Target number of abstracts per topic.")
    parser.add_argument("--output", default="data/raw_pubmed.json", help="Output JSON path.")
    parser.add_argument(
        "--topic",
        action="append",
        help="Optional topic override in the form topic=query. Can be repeated.",
    )
    args = parser.parse_args()

    if not args.email:
        parser.error("Provide --email or set the NCBI_EMAIL environment variable.")

    topics = parse_topic_overrides(args.topic)
    payload = collect_corpus(
        email=args.email,
        per_topic=args.per_topic,
        topics=topics,
        api_key=args.api_key,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {payload['num_records']} records to {output_path}")
    print(f"NCBI email: {args.email}")
    print("NCBI API key: configured" if args.api_key else "NCBI API key: not set")
    for topic in topics:
        count = sum(1 for r in payload["records"] if r["topic"] == topic)
        print(f"  - {topic}: {count}")


if __name__ == "__main__":
    main()
