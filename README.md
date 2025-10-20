# zhiweidoushu

Retrieval-augmented 紫微斗数 assistant built with FastAPI, Qdrant, and Arize Phoenix monitoring.

## Quick start with Docker Compose

1. Export any secrets you need, e.g.:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
2. Build and launch the full stack:
   ```bash
   docker compose up --build
   ```
3. Services exposed:
   - FastAPI UI: http://localhost:8000
   - Qdrant dashboard/API: http://localhost:6333
   - Phoenix UI: http://localhost:6006

Volumes `./store_location` and `./qdrant_storage` are mounted for local persistence.

Stop the stack with `docker compose down`. Add `-v` if you want to wipe the persisted vector store.

## Local development

Install dependencies and run the API directly:
```bash
pip install -e .
uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```
