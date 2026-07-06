"""
Run this once (and again whenever documents change) to build the FAISS index:

    python -m app.ingest
"""
from dotenv import load_dotenv
load_dotenv()

from app.rag_engine import build_index, DATA_DIR


def main():
    print(f"Loading documents from {DATA_DIR} ...")
    vector_store = build_index()
    print(f"Index built with {vector_store.index.ntotal} chunks and saved to disk.")


if __name__ == "__main__":
    main()
