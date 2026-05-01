# Medical Infectious-Disease RAG QA System

*Note: This branch (Notebook) is for the academic submission, containing the notebook, final paper, and scoring.*

A Retrieval-Augmented Generation (RAG) system for answering infectious-disease questions using PubMed abstracts as the knowledge base. The project compares three local open-source language models through the same retrieval pipeline to evaluate answer quality, factual grounding, and hallucination risk.

**Project authors:** Christopher Protheroe and Tyler Burzenski

## Overview

This repository builds a medical question-answering pipeline over a PubMed-sourced infectious-disease corpus. The corpus contains abstracts related to:

- COVID-19 / SARS-CoV-2
- Influenza
- Tuberculosis
- HIV

The system fetches PubMed abstracts, preprocesses and chunks them, embeds the chunks with SentenceTransformers, stores the vectors in ChromaDB, retrieves relevant context for each question, and sends the same context to multiple local LLMs through Ollama.

The goal is not to provide clinical advice. The project is an NLP/RAG comparison study focused on whether free or open-source LLMs can answer medical-domain questions more safely when grounded in retrieved PubMed evidence.

## Key Features

- PubMed corpus collection through NCBI Entrez / Biopython
- Sentence-level chunking with overlap
- Vector search using ChromaDB
- SentenceTransformer embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Shared prompt template that instructs models to answer only from retrieved context
- Local inference through Ollama
- Three-model comparison:
  - `llama3.1:8b-instruct-q4_K_M`
  - `mistral:7b-instruct-q4_1`
  - `phi3:mini`
- Batch experiment runner for 10 domain-specific questions
- Streamlit demo app for interactive querying
- Optional RAGAS evaluation for faithfulness and answer relevancy

## Repository Structure

```text
rag_medical_project/
├── data/
│   ├── raw_pubmed.json              # Raw PubMed records
│   ├── processed_chunks.json        # Cleaned and chunked records
│   └── chroma_db/                   # Persisted ChromaDB vector index
├── results/
│   ├── answers.csv                  # Model responses for evaluation questions
│   └── ragas_scores.csv             # RAGAS faithfulness / relevancy scores
├── src/
│   ├── app.py                       # Streamlit demo UI
│   ├── build_index.py               # Builds ChromaDB vector store
│   ├── evaluate.py                  # Optional RAGAS evaluation
│   ├── fetch_pubmed.py              # Fetches PubMed abstracts
│   ├── preprocess.py                # Cleans and chunks abstracts
│   ├── prompt.py                    # Shared medical-safe prompt template
│   ├── retrieve.py                  # Retrieves top-k chunks from ChromaDB
│   └── run_models.py                # Runs all questions across all models
├── requirements.txt
└── README.md
```

## Dataset

The included project data contains:

- **100 PubMed abstract records**
  - 25 COVID-19 records
  - 25 influenza records
  - 25 tuberculosis records
  - 25 HIV records
- **316 processed chunks**
  - 87 COVID-19 chunks
  - 89 influenza chunks
  - 78 tuberculosis chunks
  - 62 HIV chunks

The default corpus is created by querying PubMed for infectious-disease topics and retaining abstracts that include title and abstract text.

## Architecture

```text
PubMed Entrez API
        ↓
Raw PubMed JSON
        ↓
Cleaning + sentence chunking
        ↓
SentenceTransformer embeddings
        ↓
ChromaDB vector store
        ↓
Question → top-k retrieval
        ↓
Shared grounded prompt
        ↓
Ollama-hosted LLMs
        ↓
Answers + evaluation outputs
```

## Requirements

### Software

- Python 3.10 or newer recommended
- Ollama installed and running locally
- Git, if cloning the repository

### Python Packages

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- `biopython`
- `chromadb`
- `pandas`
- `requests`
- `sentence-transformers`
- `streamlit`
- `ragas`
- `datasets`
- `langchain-ollama`
- `langchain-core`

### Ollama Models

Pull the models used in the project:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct-q4_1
ollama pull phi3:mini
```

For optional RAGAS evaluation with local Ollama embeddings, also pull:

```bash
ollama pull nomic-embed-text
```

Make sure the Ollama server is running:

```bash
ollama serve
```

By default, the code expects Ollama at:

```text
http://localhost:11434
```

You can override this with the `OLLAMA_BASE_URL` environment variable.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/tburz/Medical-RAG-QA-System.git
cd rag_medical_project
python -m venv .venv
```

Activate the environment.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start with Included Data

The ZIP version of this project already includes `data/raw_pubmed.json`, `data/processed_chunks.json`, `data/chroma_db/`, and result CSV files. After installing dependencies and pulling the Ollama models, you can run the Streamlit demo immediately:

```bash
streamlit run src/app.py
```

Then open the local Streamlit URL shown in the terminal. The app lets you enter a medical question, inspect retrieved PubMed chunks, and compare answers from the selected local models.

## Rebuild the Pipeline from Scratch

Use this sequence if you want to regenerate the corpus, chunks, vector index, answers, and evaluation outputs.

### 1. Configure NCBI PubMed Access

NCBI Entrez requires an email address. An API key is optional but recommended.

macOS/Linux:

```bash
export NCBI_EMAIL="yourname@example.com"
export NCBI_API_KEY="YOUR_NCBI_API_KEY"
```

Windows PowerShell:

```powershell
$env:NCBI_EMAIL="yourname@example.com"
$env:NCBI_API_KEY="YOUR_NCBI_API_KEY"
```

You can also pass these values directly with `--email` and `--api-key`.

### 2. Fetch PubMed Abstracts

```bash
python src/fetch_pubmed.py \
  --email yourname@example.com \
  --per-topic 25 \
  --output data/raw_pubmed.json
```

Optional API key:

```bash
python src/fetch_pubmed.py \
  --email yourname@example.com \
  --api-key YOUR_NCBI_API_KEY \
  --per-topic 25 \
  --output data/raw_pubmed.json
```

### 3. Preprocess and Chunk the Abstracts

```bash
python src/preprocess.py \
  --input data/raw_pubmed.json \
  --output data/processed_chunks.json \
  --chunk-size 4 \
  --overlap 1
```

### 4. Build the ChromaDB Index

```bash
python src/build_index.py \
  --input data/processed_chunks.json \
  --persist-dir data/chroma_db \
  --collection pubmed_medical_rag \
  --reset
```

### 5. Test Retrieval

```bash
python src/retrieve.py "What does the corpus say about influenza vaccine effectiveness?" --top-k 3
```

You can filter retrieval by topic:

```bash
python src/retrieve.py "What diagnostic approaches are mentioned for tuberculosis?" \
  --topic-filter tuberculosis \
  --top-k 3
```

Valid topic filters in the included corpus are:

```text
covid19, influenza, tuberculosis, hiv
```

### 6. Run the Model Comparison Experiment

Make sure Ollama is running and the three model names are available locally. Then run:

```bash
python src/run_models.py --output results/answers.csv
```

This runs the default set of 10 domain-specific questions against all three models and writes 30 rows to `results/answers.csv`.

To change retrieval depth:

```bash
python src/run_models.py --top-k 5 --output results/answers.csv
```

To run a subset of models:

```bash
python src/run_models.py \
  --models phi3:mini mistral:7b-instruct-q4_1 \
  --output results/answers.csv
```

## Custom Evaluation Questions

`src/run_models.py` includes 10 default questions. You can provide your own JSON file with either a list of questions:

```json
[
  "What does the corpus say about tuberculosis diagnosis?",
  "What prevention strategies are discussed for HIV?"
]
```

Or an object with a `questions` field:

```json
{
  "questions": [
    "What complications are associated with severe COVID-19 cases?",
    "What does the corpus say about influenza severity risk factors?"
  ]
}
```

Run with:

```bash
python src/run_models.py --questions questions.json --output results/answers.csv
```

## Evaluation

The project evaluates model outputs along two main paths:

1. **Automated RAGAS scoring**
   - Faithfulness
   - Answer relevancy
2. **Manual review dimensions**
   - Consistency with retrieved PubMed context
   - Relevance to the question
   - Use of retrieved context
   - Hallucination risk
   - Handling of medical terminology

To run the optional RAGAS workflow:

```bash
python src/evaluate.py \
  --answers results/answers.csv \
  --ragas-output results/ragas_scores.csv \
  --try-ragas
```

The included `results/ragas_scores.csv` contains automated scores for the generated model answers.

## Streamlit Demo

Run:

```bash
streamlit run src/app.py
```

The app provides:

- A question input box
- Retrieval settings for Chroma directory, collection, embedding model, top-k, and topic filter
- Model selection across Llama, Mistral, and Phi
- Retrieved PubMed context inspection
- Side-by-side model outputs
- The exact prompt sent to each model
- Raw retrieval JSON

## Prompting Strategy

The shared prompt is defined in `src/prompt.py`. It instructs each model to:

- Answer only from the retrieved context
- State clearly when the evidence is insufficient
- Avoid unsupported medical claims
- Avoid fabricating statistics, diagnoses, or treatment recommendations
- Mention the study topic where possible

This keeps the model comparison fair because each LLM receives the same retrieved context and the same task instructions.

## Default Experiment Questions

The default questions in `src/run_models.py` are:

1. What evidence in the corpus describes common symptoms of COVID-19?
2. What does the corpus say about influenza vaccine effectiveness?
3. How is latent tuberculosis different from active tuberculosis?
4. What treatments are discussed for HIV infection?
5. What complications are associated with severe COVID-19 cases?
6. What diagnostic approaches are mentioned for tuberculosis?
7. What risk factors are associated with influenza severity?
8. What does the corpus say about antiviral therapy for HIV?
9. How do the retrieved studies describe prevention strategies for tuberculosis spread?
10. What limitations or uncertainties are reported in the corpus for infectious disease treatment?

## Results Files

### `results/answers.csv`

Contains one row per question/model pair:

- `question`
- `topic_filter`
- `model`
- `latency_seconds`
- `answer`
- `error`
- `retrieved_context`

### `results/ragas_scores.csv`

Contains RAGAS outputs:

- `user_input`
- `retrieved_contexts`
- `response`
- `faithfulness`
- `answer_relevancy`

## Common Troubleshooting

### Ollama connection errors

Confirm Ollama is running:

```bash
ollama serve
```

Confirm the models are installed:

```bash
ollama list
```

If Ollama is running somewhere other than `localhost:11434`, set:

```bash
export OLLAMA_BASE_URL="http://your-host:11434"
```

### ChromaDB collection not found

Rebuild the index:

```bash
python src/build_index.py --input data/processed_chunks.json --persist-dir data/chroma_db --reset
```

### Missing NCBI email

Pass `--email` to `src/fetch_pubmed.py` or set `NCBI_EMAIL` in your environment.

### RAGAS import or runtime issues

RAGAS and LangChain packages can be version-sensitive. The core RAG pipeline does not require RAGAS. If `--try-ragas` fails, the retrieval and model comparison scripts can still run normally.

### Slow model responses

Local response time depends on hardware, model size, quantization, and available memory. For faster iteration, use `phi3:mini`, reduce `--top-k`, or run fewer models with `--models`.

## Medical Disclaimer

This project is for educational and research purposes only. It is not a diagnostic tool, treatment recommendation system, or substitute for professional medical judgment. Model outputs should be interpreted as experimental NLP results grounded in the retrieved PubMed snippets, not as clinical guidance.
