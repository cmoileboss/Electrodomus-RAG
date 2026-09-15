import json as json_lib

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []
    limit: int = 5


class ChatResponse(BaseModel):
    answer: str
    history: list[Message]
    sources: list[dict]


def get_chat_service(request: Request) -> ChatService:
    return ChatService(request.app.state.embed_model)

@router.post("/chat", response_model=ChatResponse)
async def ask(body: ChatRequest, chat_service: ChatService = Depends(get_chat_service)):
    """Retourne une réponse RAG complète à partir de la question et de l'historique."""
    result = await chat_service.ask(body.question, body.history, body.limit)
    updated_history = list(body.history) + [
        Message(role="user", content=body.question),
        Message(role="assistant", content=result["answer"]),
    ]
    return ChatResponse(answer=result["answer"], history=updated_history, sources=result["sources"])


@router.post("/chat/stream")
async def ask_stream(body: ChatRequest, chat_service: ChatService = Depends(get_chat_service)):
    """Stream la réponse RAG token par token via Server-Sent Events."""
    async def generate():
        async for event in chat_service.stream_answer(body.question, body.history, body.limit):
            yield f"data: {json_lib.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
