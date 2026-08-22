from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.chunk_repository import ChunkRepository
from database.database import get_session
from database.models import Chunk

router = APIRouter(prefix="/chunks", tags=["chunks"])


class ChunkCreate(BaseModel):
    document_id: int
    chunk_index: int
    content: str
    embedding_text: str
    section: str | None = None
    page: int | None = None
    embedding: list[float] | None = None


class ChunkUpdate(BaseModel):
    content: str | None = None
    embedding_text: str | None = None
    section: str | None = None
    page: int | None = None
    embedding: list[float] | None = None


class ChunkResponse(BaseModel):
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
    with get_session() as session:
        chunks = session.query(Chunk).all()
        return [ChunkResponse.model_validate(c) for c in chunks]


@router.get("/{chunk_id}", response_model=ChunkResponse)
def get_by_id(chunk_id: int):
    with get_session() as session:
        chunk = ChunkRepository(session).get_by_id(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk introuvable")
        return ChunkResponse.model_validate(chunk)


@router.post("/", response_model=ChunkResponse, status_code=201)
def create(body: ChunkCreate):
    with get_session() as session:
        chunk = ChunkRepository(session).create(
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
    with get_session() as session:
        chunk = ChunkRepository(session).get_by_id(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk introuvable")
        for field, value in body.model_dump(exclude_unset=True).items():
            setattr(chunk, field, value)
        session.commit()
        session.refresh(chunk)
        return ChunkResponse.model_validate(chunk)


@router.delete("/{chunk_id}", status_code=204)
def delete(chunk_id: int):
    with get_session() as session:
        chunk = ChunkRepository(session).get_by_id(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk introuvable")
        session.delete(chunk)
        session.commit()
