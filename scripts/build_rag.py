"""
build_rag.py

Creates FAISS vector database.
"""

from pathlib import Path

from backend.config import RAG_DIR

from backend.rag import (

    load_pdf_folder,

    chunk_text,

    EmbeddingModel,

    FaissIndex

)


def main():

    print("=" * 70)
    print("BUILDING RAG DATABASE")
    print("=" * 70)

    text = load_pdf_folder(

        RAG_DIR / "documents"

    )

    chunks = chunk_text(text)

    embedder = EmbeddingModel()

    embeddings = embedder.encode(chunks)

    faiss_db = FaissIndex()

    faiss_db.build(embeddings)

    faiss_db.save(

        RAG_DIR / "index.faiss",

        RAG_DIR / "chunks.pkl",

        chunks

    )

    print()

    print("RAG Database Created Successfully")


if __name__ == "__main__":

    main()
