"""Endpoint /chat : reçoit une question et renvoie une réponse RAG avec ses sources."""

import json as json_lib

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


class Message(BaseModel):
    """Un message de la conversation (utilisateur ou assistant)."""

    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """Corps de la requête POST /chat."""

    question: str
    history: list[Message] = []
    limit: int = 5


class ChatResponse(BaseModel):
    """Réponse RAG : texte généré, historique mis à jour et sources citées."""

    answer: str
    history: list[Message]
    sources: list[dict]


def get_chat_service(request: Request) -> ChatService:
    """Dépendance FastAPI fournissant un ChatService lié aux modèles chargés au démarrage de l'application."""
    return ChatService(
        embed_model=request.app.state.embed_model,
        reranker=request.app.state.reranker,
    )

@router.post("/chat", response_model=ChatResponse)
async def ask(body: ChatRequest, chat_service: ChatService = Depends(get_chat_service)):
    """Retourne une réponse RAG complète à partir de la question et de l'historique."""
    result = await chat_service.ask(body.question, body.history)
    updated_history = list(body.history) + [
        Message(role="user", content=body.question),
        Message(role="assistant", content=result["answer"]),
    ]
    return ChatResponse(answer=result["answer"], history=updated_history, sources=result["sources"])
