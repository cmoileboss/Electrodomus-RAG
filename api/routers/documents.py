from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.database import get_session
from database.document_repository import DocumentRepository

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentCreate(BaseModel):
    title: str
    filepath: str
    date: datetime | None = None


class DocumentUpdate(BaseModel):
    title: str | None = None
    filepath: str | None = None
    date: datetime | None = None


class DocumentResponse(BaseModel):
    id: int
    title: str
    filepath: str
    date: datetime | None

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[DocumentResponse])
def get_all():
    with get_session() as session:
        docs = DocumentRepository(session).get_all()
        return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_by_id(document_id: int):
    with get_session() as session:
        doc = DocumentRepository(session).get_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document introuvable")
        return DocumentResponse.model_validate(doc)


@router.post("/", response_model=DocumentResponse, status_code=201)
def create(body: DocumentCreate):
    with get_session() as session:
        doc = DocumentRepository(session).create(
            title=body.title, filepath=body.filepath, date=body.date
        )
        return DocumentResponse.model_validate(doc)


@router.put("/{document_id}", response_model=DocumentResponse)
def update_by_id(document_id: int, body: DocumentUpdate):
    with get_session() as session:
        repo = DocumentRepository(session)
        doc = repo.get_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document introuvable")
        for field, value in body.model_dump(exclude_unset=True).items():
            setattr(doc, field, value)
        session.commit()
        session.refresh(doc)
        return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
def delete(document_id: int):
    with get_session() as session:
        deleted = DocumentRepository(session).delete(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document introuvable")
