"""
Creates embeddings for text chunks.
"""

from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def encode(self, texts):

        return self.model.encode(

            texts,

            convert_to_numpy=True

        )
