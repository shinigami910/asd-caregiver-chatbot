"""
test.py

Test the chatbot.
"""

from pathlib import Path

from backend.inference import (

    ASDModel,

    ResponseGenerator

)

from backend.rag import (

    EmbeddingModel,

    FaissIndex,

    Retriever

)


def main():

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

    generator = ResponseGenerator(

        model.model,

        model.tokenizer,

        retriever

    )

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":

            break

        answer = generator.generate(question)

        print()

        print(answer)


if __name__ == "__main__":

    main()
