"""
Retrieve relevant chunks from FAISS.
"""

import numpy as np

from backend.config import TOP_K


class Retriever:

    def __init__(

        self,

        embedding_model,

        faiss_index

    ):

        self.embedding_model = embedding_model

        self.faiss = faiss_index

    def retrieve(

        self,

        query,

        top_k=TOP_K

    ):

        query_embedding = self.embedding_model.encode(

            [query]

        )

        _, indices = self.faiss.index.search(

            np.array(query_embedding),

            top_k

        )

        chunks = [

            self.faiss.chunks[i]

            for i in indices[0]

        ]

        return "\n\n".join(chunks)
