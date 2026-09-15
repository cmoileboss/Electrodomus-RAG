from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from api.services.document_service import DocumentService
from api.services.ingestion_service import IngestionService
from database.database import get_session
from logger import get_logger

logger = get_logger(__name__)

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


class IngestRequest(BaseModel):
    filepath: str


@router.get("/", response_model=list[DocumentResponse])
def get_all():
    with get_session() as session:
        docs = DocumentService(session).get_all()
        return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_by_id(document_id: int):
    with get_session() as session:
        doc = DocumentService(session).get_by_id(document_id)
        if not doc:
            logger.warning("Document introuvable : %s", document_id)
            raise HTTPException(status_code=404, detail="Document introuvable")
        return DocumentResponse.model_validate(doc)


@router.post("/", response_model=DocumentResponse, status_code=201)
def create(body: DocumentCreate):
    with get_session() as session:
        doc = DocumentService(session).create(
            title=body.title, filepath=body.filepath, date=body.date
        )
        return DocumentResponse.model_validate(doc)


@router.put("/{document_id}", response_model=DocumentResponse)
def update_by_id(document_id: int, body: DocumentUpdate):
    with get_session() as session:
        doc = DocumentService(session).update(document_id, body.model_dump(exclude_unset=True))
        if not doc:
            raise HTTPException(status_code=404, detail="Document introuvable")
        return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
def delete(document_id: int):
    with get_session() as session:
        deleted = DocumentService(session).delete(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document introuvable")
    logger.info("Document %d supprimé", document_id)


@router.post("/ingest")
def ingest_single(body: IngestRequest):
    try:
        IngestionService().ingest_single(body.filepath)
    except FileNotFoundError:
        logger.warning("Fichier introuvable pour l'ingestion : %s", body.filepath)
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {body.filepath}")
    return {"success": True, "filepath": body.filepath}


@router.post("/ingest-all")
def ingest_all(background_tasks: BackgroundTasks):
    IngestionService().ingest_all(background_tasks)
    return {"success": True, "message": "Ingestion démarrée en arrière-plan"}
