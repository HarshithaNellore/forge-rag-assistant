"""
Core RAG engine for the Startup Founder Knowledge Assistant.

Pipeline:
    documents (markdown) -> chunking -> Gemini embeddings -> FAISS vector store
    query -> embed query -> retrieve top-k chunks -> prompt Gemini -> grounded answer

This module is intentionally provider-agnostic at the embedding/LLM layer:
swap GEMINI_* calls for OpenAI/Anthropic equivalents by changing only
`get_embeddings()` and `get_llm()`.
"""

import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "docs"
INDEX_DIR = BASE_DIR / "data" / "faiss_index"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4

SYSTEM_PROMPT = """You are the Founder Knowledge Assistant. Answer the user's question using ONLY
the context passages provided below. If the context does not contain enough information to answer,
say so clearly instead of guessing.

Be concise, practical, and specific. When useful, refer to which idea the advice comes from
(e.g. "idea validation", "fundraising"), but do not fabricate sources.

Context:
{context}

Question: {question}

Answer:"""


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Returns the embedding model. Requires GOOGLE_API_KEY in the environment."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_llm() -> ChatGoogleGenerativeAI:
    """Returns the chat model used for answer generation."""
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

def load_documents() -> List[Document]:
    loader = DirectoryLoader(str(DATA_DIR), glob="**/*.md", loader_cls=TextLoader)
    return loader.load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index() -> FAISS:
    """Loads documents, chunks them, embeds them, and builds/saves a FAISS index."""
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {DATA_DIR}")
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store


def load_index() -> FAISS:
    """Loads a previously built FAISS index from disk."""
    if not INDEX_DIR.exists():
        raise RuntimeError("No FAISS index found. Run `python -m app.ingest` first.")
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


def retrieve(vector_store: FAISS, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    return vector_store.similarity_search_with_score(query, k=k)


def answer_question(vector_store: FAISS, question: str) -> dict:
    """Runs full retrieval + generation and returns the answer with sources."""
    results = retrieve(vector_store, question)
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    llm = get_llm()
    response = llm.invoke(prompt)
    raw_content = response.content
    if isinstance(raw_content, str):
        answer_text = raw_content
    elif isinstance(raw_content, list):
        answer_text = "".join(
            block.get("text", "") for block in raw_content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        answer_text = str(raw_content)

    sources = sorted({
        Path(doc.metadata.get("source", "unknown")).name for doc, _ in results
    })

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": [
            {"text": doc.page_content, "score": float(score), "source": Path(doc.metadata.get("source", "")).name}
            for doc, score in results
        ],
    }
