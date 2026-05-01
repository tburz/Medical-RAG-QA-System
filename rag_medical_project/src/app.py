from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

from prompt import build_prompt
from retrieve import retrieve_chunks


MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_1",
    "phi3:mini",
]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def query_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("response", "")


st.set_page_config(page_title="Medical RAG Demo", layout="wide")
st.title("Medical Infectious-Disease RAG Demo")
st.write("Ask a question, inspect the retrieved PubMed chunks, and compare local open-source models.")

with st.sidebar:
    st.header("Settings")
    persist_dir = st.text_input("Chroma persist dir", value="data/chroma_db")
    collection_name = st.text_input("Collection name", value="pubmed_medical_rag")
    embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    top_k = st.slider("Top-k retrieval", min_value=1, max_value=8, value=3)
    selected_models = st.multiselect("Models", MODELS, default=MODELS)
    topic_filter = st.selectbox("Topic filter", ["", "covid19", "influenza", "tuberculosis", "hiv"])

question = st.text_area(
    "Question",
    value="What does the corpus say about influenza vaccine effectiveness?",
    height=100,
)

if st.button("Run RAG"):
    with st.spinner("Retrieving context..."):
        retrieved = retrieve_chunks(
            query=question,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            top_k=top_k,
            topic_filter=topic_filter or None,
        )
        prompt = build_prompt(question, retrieved)

    st.subheader("Retrieved context")
    for i, chunk in enumerate(retrieved, start=1):
        with st.expander(f"Chunk {i}: {chunk['metadata'].get('title', 'Untitled')}"):
            st.json(chunk["metadata"])
            st.write(chunk["document"])

    st.subheader("Model outputs")
    cols = st.columns(len(selected_models)) if selected_models else []
    for col, model in zip(cols, selected_models):
        with col:
            st.markdown(f"### {model}")
            with st.spinner(f"Querying {model}..."):
                try:
                    answer = query_ollama(model, prompt)
                    st.write(answer)
                except Exception as exc:
                    st.error(str(exc))

    with st.expander("Prompt used"):
        st.code(prompt)

    with st.expander("Raw retrieval JSON"):
        st.code(json.dumps(retrieved, indent=2, ensure_ascii=False))
