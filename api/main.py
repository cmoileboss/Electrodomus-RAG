import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from database.chunk_repository import ChunkRepository
from database.database import get_session, init_db
from doc_management.chunks_storing import process_all, process_single
from logger import get_logger
from ollama_client import SYSTEM_PROMPT, build_rag_message, chat

logger = get_logger(__name__)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")

embed_model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model
    init_db()
    logger.info("Initialisation de la base de données terminée.")
    logger.info("Chargement du modèle d'embedding : %s", EMBED_MODEL_ID)
    embed_model = SentenceTransformer(EMBED_MODEL_ID, token=HF_TOKEN)
    yield
    embed_model = None


app = FastAPI(title="Electrodomus RAG API", lifespan=lifespan)


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
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def ask(request: ChatRequest):
    embedding = embed_model.encode(request.question, normalize_embeddings=True).tolist()

    with get_session() as session:
        chunks = ChunkRepository(session).get_nearest(embedding, limit=request.limit)

    rag_content = build_rag_message(request.question, chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": rag_content})

    answer = chat(messages, stream=False)

    updated_history = list(request.history) + [
        Message(role="user", content=request.question),
        Message(role="assistant", content=answer),
    ]
    sources = [
        {"section": c.section, "page": c.page, "document_id": c.document_id}
        for c in chunks
    ]
    return ChatResponse(answer=answer, history=updated_history, sources=sources)


@app.post("/documents/ingest")
def ingest_single(request: IngestRequest):
    try:
        process_single(request.filepath)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {request.filepath}")
    return {"success": True, "filepath": request.filepath}


@app.post("/documents/ingest-all")
def ingest_all():
    process_all()
    return {"success": True}
