"""
FastAPI routes.
"""

from fastapi import APIRouter

from backend.api.schemas import (

    ChatRequest,

    ChatResponse

)

router = APIRouter()


# This object will be initialized in app.py
generator = None


@router.get("/health")

def health():

    return {

        "status": "healthy"

    }


@router.post(

    "/chat",

    response_model=ChatResponse

)

def chat(request: ChatRequest):

    answer = generator.generate(

        request.question

    )

    return ChatResponse(

        response=answer

    )
