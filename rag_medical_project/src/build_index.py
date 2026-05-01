from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ChromaDB index from processed chunks.")
    parser.add_argument("--input", default="data/processed_chunks.json")
    parser.add_argument("--persist-dir", default="data/chroma_db")
    parser.add_argument("--collection", default="pubmed_medical_rag")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--reset", action="store_true", help="Delete the existing ChromaDB directory first.")
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    if args.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise ValueError("No chunks found in input file.")

    client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.embedding_model)

    existing = {c.name for c in client.list_collections()}
    if args.collection in existing:
        client.delete_collection(args.collection)

    collection = client.create_collection(name=args.collection, embedding_function=embedding_fn)

    ids = [item["id"] for item in chunks]
    docs = [item["text"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]

    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(ids=ids[start:end], documents=docs[start:end], metadatas=metadatas[start:end])

    print(f"Indexed {len(chunks)} chunks into collection '{args.collection}' at {persist_dir}")


if __name__ == "__main__":
    main()
