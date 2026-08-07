"""
Create, save and load FAISS index.
"""

import pickle
from pathlib import Path

import faiss


class FaissIndex:

    def __init__(self):

        self.index = None

        self.chunks = None

    def build(self, embeddings):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

    def save(

        self,

        index_path,

        chunks_path,

        chunks

    ):

        faiss.write_index(

            self.index,

            str(index_path)

        )

        with open(chunks_path, "wb") as f:

            pickle.dump(chunks, f)

    def load(

        self,

        index_path,

        chunks_path

    ):

        self.index = faiss.read_index(

            str(index_path)

        )

        with open(chunks_path, "rb") as f:

            self.chunks = pickle.load(f)
