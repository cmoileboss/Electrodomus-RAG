"""Endpoints CRUD /documents et endpoints d'ingestion de documents."""

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.services.document_service import DocumentService
from api.services.ingestion_service import IngestionService
from database.database import get_session
from doc_management.chunks_storing import DOC_FOLDER, SUPPORTED_EXTENSIONS
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentCreate(BaseModel):
    """Données requises et facultatives pour créer un document.

    Attributes:
        title: Titre unique du document.
        filepath: Chemin du fichier source.
        date: Date du document, si elle est connue.
    """

    title: str
    filepath: str
    date: datetime | None = None


class DocumentUpdate(BaseModel):
    """Champs modifiables d'un document.

    Attributes:
        title: Nouveau titre du document.
        filepath: Nouveau chemin du fichier source.
        date: Nouvelle date du document.
    """

    title: str | None = None
    filepath: str | None = None
    date: datetime | None = None


class DocumentResponse(BaseModel):
    """Représentation publique d'un document.

    Attributes:
        id: Identifiant du document en base.
        title: Titre unique du document.
        filepath: Chemin du fichier source associé.
        date: Date du document, si elle est connue.
    """

    id: int
    title: str
    filepath: str
    date: datetime | None

    model_config = {"from_attributes": True}


class IngestRequest(BaseModel):
    """Données nécessaires à l'ingestion d'un fichier existant.

    Attributes:
        filepath: Chemin du document à convertir et indexer.
    """

    filepath: str


@router.get("/", response_model=list[DocumentResponse])
def get_all():
    """Liste tous les documents enregistrés.

    Returns:
        list[DocumentResponse]: Représentations API des documents présents en base.
    """
    with get_session() as session:
        docs = DocumentService(session).get_all()
        return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_by_id(document_id: int):
    """Récupère un document par son identifiant.

    Args:
        document_id: Identifiant du document demandé.

    Returns:
        DocumentResponse: Représentation du document trouvé.

    Raises:
        HTTPException: Code 404 lorsque le document n'existe pas.
    """
    with get_session() as session:
        doc = DocumentService(session).get_by_id(document_id)
        if not doc:
            logger.warning("Document introuvable : %s", document_id)
            raise HTTPException(status_code=404, detail="Document introuvable")
        return DocumentResponse.model_validate(doc)


@router.post("/", response_model=DocumentResponse, status_code=201)
def create(body: DocumentCreate):
    """Crée un document à partir des données reçues.

    Args:
        body: Titre, chemin source et date éventuelle du document.

    Returns:
        DocumentResponse: Document créé, avec son identifiant en base.
    """
    with get_session() as session:
        doc = DocumentService(session).create(
            title=body.title, filepath=body.filepath, date=body.date
        )
        return DocumentResponse.model_validate(doc)


@router.put("/{document_id}", response_model=DocumentResponse)
def update_by_id(document_id: int, body: DocumentUpdate):
    """Modifie les champs transmis d'un document existant.

    Args:
        document_id: Identifiant du document à modifier.
        body: Champs modifiés; seuls ceux transmis sont appliqués.

    Returns:
        DocumentResponse: Document actualisé.

    Raises:
        HTTPException: Code 404 lorsque le document n'existe pas.
    """
    with get_session() as session:
        doc = DocumentService(session).update(document_id, body.model_dump(exclude_unset=True))
        if not doc:
            raise HTTPException(status_code=404, detail="Document introuvable")
        return DocumentResponse.model_validate(doc)


@router.delete("/all", status_code=200)
def delete_all():
    """Supprime tous les documents enregistrés et leurs chunks associés.

    Returns:
        dict[str, bool | int]: Résultat de l'opération et nombre de documents supprimés.
    """
    with get_session() as session:
        count = DocumentService(session).delete_all()
    logger.info("%d document(s) supprimé(s)", count)
    return {"success": True, "deleted": count}


@router.delete("/{document_id}", status_code=204)
def delete(document_id: int):
    """Supprime un document par son identifiant.

    Args:
        document_id: Identifiant du document à supprimer.

    Returns:
        None: Réponse HTTP sans contenu (204) en cas de suppression.

    Raises:
        HTTPException: Code 404 lorsque le document n'existe pas.
    """
    with get_session() as session:
        deleted = DocumentService(session).delete(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document introuvable")
    logger.info("Document %d supprimé", document_id)


@router.post("/ingest")
def ingest_single(body: IngestRequest):
    """Lance l'ingestion synchrone d'un fichier.

    Args:
        body: Requête contenant le chemin du fichier à ingérer.

    Returns:
        dict[str, bool | str]: Confirmation et chemin du fichier ingéré.

    Raises:
        HTTPException: Code 404 lorsque le fichier source est introuvable.
    """
    try:
        IngestionService().ingest_single(body.filepath)
    except FileNotFoundError:
        logger.warning("Fichier introuvable pour l'ingestion : %s", body.filepath)
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {body.filepath}")
    return {"success": True, "filepath": body.filepath}


@router.post("/ingest-all")
def ingest_all(background_tasks: BackgroundTasks):
    """Planifie l'ingestion de tous les documents du dossier source.

    Args:
        background_tasks: Gestionnaire FastAPI chargé de lancer l'ingestion après
            la réponse HTTP.

    Returns:
        dict[str, bool | str]: Confirmation que la tâche d'ingestion a été planifiée.
    """
    IngestionService().ingest_all(background_tasks)
    return {"success": True, "message": "Ingestion démarrée en arrière-plan"}


@router.post("/ingest-upload")
async def ingest_upload(file: UploadFile = File(...)):
    """Enregistre un fichier téléversé puis lance son ingestion.

    Args:
        file: Fichier transmis par le client; son extension doit être prise en charge.

    Returns:
        dict[str, bool | str]: Confirmation et chemin du fichier enregistré.

    Raises:
        HTTPException: Code 400 si l'extension n'est pas prise en charge, ou 404 si
            le fichier n'est plus accessible lors de l'ingestion.
    """
    # Path(...).name élimine tout composant de dossier pour éviter une traversée de chemin.
    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        logger.info("Extension non supportée pour l'ingestion : %s", extension)
        raise HTTPException(status_code=400, detail=f"Extension non supportée : {extension}")

    DOC_FOLDER.mkdir(parents=True, exist_ok=True)
    destination = DOC_FOLDER / filename
    with destination.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        IngestionService().ingest_single(str(destination))
    except FileNotFoundError:
        logger.warning("Fichier introuvable après enregistrement : %s", destination)
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {destination}")
    return {"success": True, "filepath": str(destination)}
