# Founder Knowledge Assistant (RAG)

A small end-to-end Retrieval-Augmented Generation (RAG) API that answers startup/founder questions
(idea validation, MVPs, fundraising, hiring, go-to-market) grounded in a curated knowledge base —
built as a practical exercise in the same stack used for LLM-powered product features: LangChain,
FAISS, prompt engineering, and a FastAPI service layer.

## Architecture

```
data/docs/*.md  --chunk-->  Gemini Embeddings  --index-->  FAISS vector store
                                                                    |
user question --embed--> similarity search (top-k) --> context --> Gemini (gemini-1.5-flash)
                                                                    |
                                                              grounded answer + sources
```

- **Chunking**: `RecursiveCharacterTextSplitter`, 800 chars/chunk with 120-char overlap, splitting
  on markdown headers first so chunks stay topically coherent.
- **Embeddings**: `GoogleGenerativeAIEmbeddings` (Gemini `embedding-001`).
- **Vector store**: FAISS, persisted to disk under `data/faiss_index/` so re-embedding isn't needed
  on every restart.
- **Generation**: `gemini-1.5-flash`, low temperature, prompted to answer only from retrieved
  context and say so explicitly when the context is insufficient (reduces hallucination).
- **API layer**: FastAPI with `/ask`, `/health`, and `/reindex` endpoints.

## Setup

```bash
git clone <this-repo>
cd forge-rag-assistant
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your key from https://aistudio.google.com/app/apikey
```

## Build the index and run

```bash
python -m app.ingest        # builds the FAISS index from data/docs/*.md
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I validate a startup idea before building anything?"}'
```

## Extending this project

- **Add documents**: drop more `.md`/`.txt` files into `data/docs/` and call `POST /reindex` (or
  re-run `python -m app.ingest`).
- **Swap the LLM provider**: only `get_embeddings()` and `get_llm()` in `app/rag_engine.py` need to
  change — the rest of the pipeline is provider-agnostic.
- **Swap vector stores**: FAISS is local and free; swapping in Pinecone/Weaviate/Chroma only
  touches `build_index()`/`load_index()`.
- **Agentic layer**: this is intentionally a single-hop RAG pipeline. A natural next step is adding
  a LangGraph agent on top that decides *whether* to retrieve, reformulates the query, or chains
  multiple retrieval steps for multi-part questions.
- **Evaluation**: a good next addition is a small eval set of (question, expected-source) pairs to
  measure retrieval precision/recall as the knowledge base grows.

## Why this project

Built to demonstrate the core building blocks behind LLM-powered product features — RAG pipeline
design, prompt engineering, vector search, and a production-style API — applied to a founder/startup
knowledge domain.
