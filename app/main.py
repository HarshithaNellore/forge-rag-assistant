"""
FastAPI service for the Startup Founder Knowledge Assistant.

Run with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag_engine import load_index, answer_question, build_index, INDEX_DIR

vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    try:
        vector_store = load_index()
        print("Loaded existing FAISS index.")
    except RuntimeError:
        print("No index found. Building one now from data/docs ...")
        vector_store = build_index()
        print("Index built.")
    yield


app = FastAPI(
    title="Founder Knowledge Assistant",
    description="A RAG-powered API answering startup/founder questions, grounded in a curated knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": vector_store is not None}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Index not ready yet.")

    result = answer_question(vector_store, request.question)
    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.post("/reindex")
def reindex():
    """Rebuilds the FAISS index from the current contents of data/docs."""
    global vector_store
    vector_store = build_index()
    return {"status": "reindexed", "chunks": vector_store.index.ntotal}
