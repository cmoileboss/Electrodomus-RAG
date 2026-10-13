"""Endpoint /chat : reçoit une question et renvoie une réponse RAG avec ses sources."""

import json as json_lib

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


class Message(BaseModel):
    """Message échangé dans l'historique de conversation.

    Attributes:
        role: Rôle émetteur du message, généralement ``user`` ou ``assistant``.
        content: Contenu textuel du message.
    """

    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """Paramètres acceptés par l'endpoint ``POST /chat``.

    Attributes:
        question: Question à traiter avec le pipeline RAG.
        history: Messages précédents servant de contexte conversationnel.
        limit: Nombre de résultats contextuels demandé; actuellement non appliqué
            par le service de chat.
        model_id: Identifiant facultatif du modèle d'appareil à filtrer.
        error_code: Code d'erreur facultatif à utiliser comme filtre de recherche.
    """

    question: str
    history: list[Message] = []
    limit: int = 5
    model_id: int | None = None
    error_code: str | None = None


class ChatResponse(BaseModel):
    """Réponse de l'endpoint de chat avec l'historique et les sources.

    Attributes:
        answer: Texte généré par le modèle de langage.
        history: Historique d'entrée auquel ont été ajoutés question et réponse.
        sources: Métadonnées des chunks utilisés pour produire la réponse.
    """

    answer: str
    history: list[Message]
    sources: list[dict]


def get_chat_service(request: Request) -> ChatService:
    """Construit le service de chat avec les modèles chargés par l'application.

    Args:
        request: Requête FastAPI permettant d'accéder à l'état applicatif.

    Returns:
        ChatService: Service configuré avec l'embedder et le reranker partagés.
    """
    return ChatService(
        embed_model=request.app.state.embed_model,
        reranker=request.app.state.reranker,
    )

@router.post("/chat", response_model=ChatResponse)
async def ask(body: ChatRequest, chat_service: ChatService = Depends(get_chat_service)):
    """Répond à une question et ajoute l'échange à l'historique.

    Args:
        body: Question, filtres facultatifs et historique fournis par le client.
        chat_service: Service RAG injecté par FastAPI.

    Returns:
        ChatResponse: Réponse générée, historique actualisé et sources citées.
    """
    result = await chat_service.ask(body.question, body.history, model_id=body.model_id, error_code=body.error_code)
    updated_history = list(body.history) + [
        Message(role="user", content=body.question),
        Message(role="assistant", content=result["answer"]),
    ]
    return ChatResponse(answer=result["answer"], history=updated_history, sources=result["sources"])
