from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_collection(persist_dir: str, collection_name: str, embedding_model: str):
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return client.get_collection(name=collection_name, embedding_function=embedding_fn)


def retrieve_chunks(
    query: str,
    persist_dir: str = "data/chroma_db",
    collection_name: str = "pubmed_medical_rag",
    embedding_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = 3,
    topic_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    collection = get_collection(persist_dir, collection_name, embedding_model)
    where = {"topic": topic_filter} if topic_filter else None
    results = collection.query(query_texts=[query], n_results=top_k, where=where)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    output: List[Dict[str, Any]] = []
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        output.append(
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks from ChromaDB.")
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("--persist-dir", default="data/chroma_db")
    parser.add_argument("--collection", default="pubmed_medical_rag")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--topic-filter")
    args = parser.parse_args()

    results = retrieve_chunks(
        query=args.query,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        topic_filter=args.topic_filter,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
