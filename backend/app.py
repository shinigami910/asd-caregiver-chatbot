"""
app.py

Application entry point.
"""

from fastapi import FastAPI

from backend.api.routes import (

    router,

    generator

)

from backend.inference import (

    ASDModel,

    ResponseGenerator

)

from backend.rag import (

    EmbeddingModel,

    FaissIndex,

    Retriever

)

from pathlib import Path

app = FastAPI(

    title="ASD Caregiver Assistant"

)

#
# Load everything once
#

model = ASDModel().load()

embedder = EmbeddingModel()

faiss_db = FaissIndex()

faiss_db.load(

    Path("rag/index.faiss"),

    Path("rag/chunks.pkl")

)

retriever = Retriever(

    embedder,

    faiss_db

)

#
# Share generator with routes
#

import backend.api.routes as routes

routes.generator = ResponseGenerator(

    model.model,

    model.tokenizer,

    retriever

)

app.include_router(router)
