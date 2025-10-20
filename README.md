# zhiweidoushu

Retrieval‑augmented Zi Wei Dou Shu (紫微斗数) assistant built with FastAPI, LangChain, Qdrant (hybrid sparse+dense), and Arize Phoenix tracing/monitoring.
<img width="1060" height="753" alt="image" src="https://github.com/user-attachments/assets/06a70640-dabe-4a04-aba5-e40c49772234" />

This repository contains:
- A FastAPI UI (`rag_api.py`) to collect birth info + question and return a generated analysis.
- A LangChain retrieval workflow (`qdrant_workflow.py`) that: generates a chart → builds search queries → retrieves from Qdrant → reranks with Flashrank → summarizes → answers (and translates if needed).
- Notebooks for data prep and evaluation (`Chunking.ipynb`, `evaluation.ipynb`, `generate grond-truth-data.ipynb`).

Note: There is no `docker-compose.yml` in this repo. Instructions below show how to run components locally or with simple Docker commands.

## Objective

Build an end‑to‑end RAG application that answers Zi Wei Dou Shu questions by combining a user’s chart with retrieved domain knowledge, and provide documentation, evaluation, and monitoring to make the system reproducible and reviewable.

## Problem Statement

Users ask nuanced Zi Wei Dou Shu questions (career, relationships, health, etc.). The system must:
- Generate an accurate natal chart from birth details.
- Convert the question + chart context into targeted retrieval queries.
- Retrieve, rerank, summarize, and answer using an LLM.
- Provide an interface (web UI) and basic monitoring.

## Technologies

- Interface: FastAPI (simple HTML page served by `rag_api.py`).
- LLM & Orchestration: LangChain (`ChatOpenAI`, `RunnableLambda`).
- Retrieval: Qdrant with `FastEmbedEmbeddings` (dense) + `FastEmbedSparse` (BM25) hybrid search.
- Reranking: `FlashrankRerank` contextual compression.
- Monitoring: Arize Phoenix via OpenInference.

## Retrieval flow
This project expects a Qdrant collection populated with your relevant knowledge. Use the notebooks to prepare and load data:
- Hybrid search, document re‑ranking, and query rewriting are used in the retrieval process.
- `Chunking.ipynb`: chunk and index the Zi Wei Dou Shu book into Qdrant and the local `store_location` KV store used by the ParentDocumentRetriever.

## Retrieval evaluation
A ground‑truth dataset (queries) was generated from the chunks to verify whether retrieval returns the correct chunk.
- Metrics: Mean Reciprocal Rank (MRR) and Hit Rate were used to evaluate retrieval performance.
- Results: MRR = 44.6%, Hit Rate = 57.8%.
- Details: see `generate grond-truth-data.ipynb`.

## LLM evaluation
A QA dataset was generated and an LLM judge scored responses for “relevance” and “faithfulness” on a 1–5 scale.
- Faithfulness: 4.7
- Relevance: 4.9
- Details: see `generate grond-truth-data.ipynb`.

## Monitoring & Feedback
- Phoenix tracing is enabled via `phoenix.otel.register` in `qdrant_workflow.py`. Start Phoenix to view traces of retrieval and LLM calls.
- You can extend the HTML form to collect thumbs‑up/down or free‑text feedback and persist it alongside questions and answers for later analysis.

## Interface
We use FastAPI for the application interface (served by `rag_api.py`).


## Prerequisites

- Python 3.10+
- uv (package manager) — recommended. Install:
  - macOS/Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  - Windows (PowerShell):
    ```powershell
    iwr https://astral.sh/uv/install.ps1 -useb | iex
    ```
- An OpenAI API key for LLM calls (`OPENAI_API_KEY`), or provide the key in the UI form.
- A running Qdrant server (default `http://localhost:6333`).
- Optional: Arize Phoenix for tracing (default collector `http://localhost:6006`).

## Quick Start (uv)

```bash
# 1) Clone the repo
git clone https://github.com/<your-username>/ziweidoushu.git
cd ziweidoushu

# 2) Sync deps into a local .venv from uv.lock
uv sync --dev --frozen

# 3) Start Qdrant in another terminal (see below)
# 4) (Optional) Start Phoenix (see below)

# 5) Run the API
uv run uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- `uv sync --frozen` uses the locked versions in `uv.lock` for reproducibility.
- `uv run` executes in the project’s virtualenv without manual activation.

### Start Qdrant (recommended via Docker)

```bash
docker run --name qdrant -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

This persists data under `./qdrant_storage`.

### Start Phoenix (optional)

```bash
pip install arize-phoenix
phoenix serve --host 0.0.0.0 --port 6006
```

The workflow is instrumented to emit traces to `PHOENIX_COLLECTOR_ENDPOINT` (`http://localhost:6006` by default).


##  Run the FastAPI app with uv:

```bash
uv run uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```
