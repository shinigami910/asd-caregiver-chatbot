"""
Split long documents into overlapping chunks.
"""

from backend.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def chunk_text(

    text,

    chunk_size=CHUNK_SIZE,

    overlap=CHUNK_OVERLAP

):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks
