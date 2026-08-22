import asyncio
import json as json_lib
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from database.chunk_repository import ChunkRepository
from database.database import get_session, init_db
from doc_management.chunks_storing import process_all, process_single
from logger import get_logger
from ollama_client import SYSTEM_PROMPT, build_rag_message, chat, stream_chat
from api.routers import chunks as chunks_router
from api.routers import documents as documents_router
from api.routers import models as models_router

logger = get_logger(__name__)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")

embed_model: SentenceTransformer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise la base de données et charge le modèle d'embedding au démarrage."""
    global embed_model
    init_db()
    logger.info("Initialisation de la base de données terminée.")
    logger.info("Chargement du modèle d'embedding : %s", EMBED_MODEL_ID)
    embed_model = await asyncio.to_thread(
        lambda: SentenceTransformer(EMBED_MODEL_ID, use_auth_token=HF_TOKEN)
    )
    logger.info("Modèle d'embedding chargé.")
    yield


app = FastAPI(title="Electrodomus RAG API", lifespan=lifespan)

app.include_router(chunks_router.router)
app.include_router(documents_router.router)
app.include_router(models_router.router)


# --- Schémas ---

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


class IngestRequest(BaseModel):
    filepath: str


# --- Endpoints ---

@app.get("/health")
def health():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def ask(request: ChatRequest):
    """Retourne une réponse RAG complète à partir de la question et de l'historique."""
    embedding = await asyncio.to_thread(
        embed_model.encode, request.question, normalize_embeddings=True
    )

    with get_session() as session:
        chunks = ChunkRepository(session).get_nearest(embedding.tolist(), limit=request.limit)

    rag_content = build_rag_message(request.question, chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": rag_content})

    answer = await asyncio.to_thread(chat, messages, False)

    updated_history = list(request.history) + [
        Message(role="user", content=request.question),
        Message(role="assistant", content=answer),
    ]
    sources = [
        {"section": c.section, "page": c.page, "document_id": c.document_id}
        for c in chunks
    ]
    return ChatResponse(answer=answer, history=updated_history, sources=sources)


@app.post("/chat/stream")
async def ask_stream(request: ChatRequest):
    """Stream la réponse RAG token par token via Server-Sent Events."""
    embedding = await asyncio.to_thread(
        embed_model.encode, request.question, normalize_embeddings=True
    )

    with get_session() as session:
        chunks = ChunkRepository(session).get_nearest(embedding.tolist(), limit=request.limit)

    rag_content = build_rag_message(request.question, chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": rag_content})

    sources = [
        {"section": c.section, "page": c.page, "document_id": c.document_id}
        for c in chunks
    ]

    async def generate():
        yield f"data: {json_lib.dumps({'type': 'sources', 'sources': sources})}\n\n"
        full_answer = ""
        async for token in stream_chat(messages):
            full_answer += token
            yield f"data: {json_lib.dumps({'type': 'token', 'content': token})}\n\n"
        updated_history = [
            {"role": m.role, "content": m.content} for m in request.history
        ] + [
            {"role": "user", "content": request.question},
            {"role": "assistant", "content": full_answer},
        ]
        yield f"data: {json_lib.dumps({'type': 'done', 'history': updated_history})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/documents/ingest")
def ingest_single(request: IngestRequest):
    """Convertit, découpe et indexe un document unique en base."""
    try:
        process_single(request.filepath)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {request.filepath}")
    return {"success": True, "filepath": request.filepath}


@app.post("/documents/ingest-all")
def ingest_all(background_tasks: BackgroundTasks):
    """Lance l'ingestion de tous les documents du dossier Documentation_Electrodomus en arrière-plan."""
    background_tasks.add_task(process_all)
    return {"success": True, "message": "Ingestion démarrée en arrière-plan"}
