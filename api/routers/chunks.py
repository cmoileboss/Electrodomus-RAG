"""Endpoints CRUD /chunks pour gérer les chunks de documents indeixés en base."""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.chunk_service import ChunkService
from database.database import get_session
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chunks", tags=["chunks"])


class ChunkCreate(BaseModel):
    """Corps de requête pour créer un chunk."""

    document_id: int
    chunk_index: int
    content: str
    embedding_text: str
    section: str | None = None
    page: int | None = None
    embedding: list[float] | None = None


class ChunkUpdate(BaseModel):
    """Corps de requête pour mettre à jour partiellement un chunk."""

    content: str | None = None
    embedding_text: str | None = None
    section: str | None = None
    page: int | None = None
    embedding: list[float] | None = None


class ChunkResponse(BaseModel):
    """Représentation d'un chunk retournée par l'API."""

    id: int
    document_id: int
    chunk_index: int
    content: str
    embedding_text: str
    section: str | None
    page: int | None
    created_at: datetime | None

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[ChunkResponse])
def get_all():
    """Liste tous les chunks."""
    with get_session() as session:
        chunks = ChunkService(session).get_all()
        return [ChunkResponse.model_validate(c) for c in chunks]


@router.get("/count")
def count():
    """Retourne le nombre total de chunks."""
    with get_session() as session:
        return {"count": ChunkService(session).count()}


@router.get("/{chunk_id}", response_model=ChunkResponse)
def get_by_id(chunk_id: int):
    """Récupère un chunk par son id, ou 404 si introuvable."""
    with get_session() as session:
        chunk = ChunkService(session).get_by_id(chunk_id)
        if not chunk:
            logger.warning("Chunk introuvable : %s", chunk_id)
            raise HTTPException(status_code=404, detail="Chunk introuvable")
        return ChunkResponse.model_validate(chunk)


@router.post("/", response_model=ChunkResponse, status_code=201)
def create(body: ChunkCreate):
    """Crée un nouveau chunk."""
    with get_session() as session:
        chunk = ChunkService(session).create(
            document_id=body.document_id,
            chunk_index=body.chunk_index,
            content=body.content,
            embedding_text=body.embedding_text,
            section=body.section,
            page=body.page,
            embedding=body.embedding,
        )
        return ChunkResponse.model_validate(chunk)


@router.put("/{chunk_id}", response_model=ChunkResponse)
def update_by_id(chunk_id: int, body: ChunkUpdate):
    """Met à jour partiellement un chunk existant, ou 404 si introuvable."""
    with get_session() as session:
        chunk = ChunkService(session).update(chunk_id, body.model_dump(exclude_unset=True))
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk introuvable")
        return ChunkResponse.model_validate(chunk)


@router.delete("/{chunk_id}", status_code=204)
def delete(chunk_id: int):
    """Supprime un chunk par son id, ou 404 si introuvable."""
    with get_session() as session:
        deleted = ChunkService(session).delete(chunk_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chunk introuvable")
