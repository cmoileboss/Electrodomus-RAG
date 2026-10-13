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
    """Données requises et facultatives pour créer un chunk.

    Attributes:
        document_id: Identifiant du document parent.
        chunk_index: Position du chunk dans le document.
        content: Texte extrait du document.
        embedding_text: Texte contextualisé servant à calculer l'embedding.
        section: Section source, si elle est connue.
        page: Numéro de page source, si disponible.
        embedding: Vecteur associé au texte, si déjà calculé.
    """

    document_id: int
    chunk_index: int
    content: str
    embedding_text: str
    section: str | None = None
    page: int | None = None
    embedding: list[float] | None = None


class ChunkUpdate(BaseModel):
    """Champs modifiables d'un chunk.

    Attributes:
        content: Nouveau texte source.
        embedding_text: Nouveau texte contextualisé utilisé pour l'embedding.
        section: Nouvelle section source.
        page: Nouveau numéro de page source.
        embedding: Nouveau vecteur associé au chunk.
    """

    content: str | None = None
    embedding_text: str | None = None
    section: str | None = None
    page: int | None = None
    embedding: list[float] | None = None


class ChunkResponse(BaseModel):
    """Représentation publique d'un chunk et de ses métadonnées.

    Attributes:
        id: Identifiant du chunk en base.
        document_id: Identifiant du document parent.
        chunk_index: Position du chunk dans le document.
        content: Texte source du chunk.
        embedding_text: Texte contextualisé utilisé pour la recherche vectorielle.
        section: Section source, si elle est connue.
        page: Numéro de page source, si disponible.
        created_at: Date de création en base, si elle est renseignée.
    """

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
    """Liste tous les chunks enregistrés.

    Returns:
        list[ChunkResponse]: Représentations API des chunks présents en base.
    """
    with get_session() as session:
        chunks = ChunkService(session).get_all()
        return [ChunkResponse.model_validate(c) for c in chunks]


@router.get("/count")
def count():
    """Compte les chunks enregistrés.

    Returns:
        dict[str, int]: Objet ``{"count": nombre}`` consommé par l'API.
    """
    with get_session() as session:
        return {"count": ChunkService(session).count()}


@router.get("/{chunk_id}", response_model=ChunkResponse)
def get_by_id(chunk_id: int):
    """Récupère un chunk par son identifiant.

    Args:
        chunk_id: Identifiant du chunk demandé.

    Returns:
        ChunkResponse: Représentation du chunk trouvé.

    Raises:
        HTTPException: Code 404 lorsque le chunk n'existe pas.
    """
    with get_session() as session:
        chunk = ChunkService(session).get_by_id(chunk_id)
        if not chunk:
            logger.warning("Chunk introuvable : %s", chunk_id)
            raise HTTPException(status_code=404, detail="Chunk introuvable")
        return ChunkResponse.model_validate(chunk)


@router.post("/", response_model=ChunkResponse, status_code=201)
def create(body: ChunkCreate):
    """Crée un chunk à partir des données reçues.

    Args:
        body: Données du chunk à créer, avec métadonnées facultatives.

    Returns:
        ChunkResponse: Chunk créé, avec son identifiant en base.
    """
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
    """Modifie les champs fournis d'un chunk existant.

    Args:
        chunk_id: Identifiant du chunk à modifier.
        body: Champs modifiés; seuls les champs transmis sont appliqués.

    Returns:
        ChunkResponse: Chunk actualisé.

    Raises:
        HTTPException: Code 404 lorsque le chunk n'existe pas.
    """
    with get_session() as session:
        chunk = ChunkService(session).update(chunk_id, body.model_dump(exclude_unset=True))
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk introuvable")
        return ChunkResponse.model_validate(chunk)


@router.delete("/{chunk_id}", status_code=204)
def delete(chunk_id: int):
    """Supprime un chunk par son identifiant.

    Args:
        chunk_id: Identifiant du chunk à supprimer.

    Returns:
        None: Réponse HTTP sans contenu (204) en cas de suppression.

    Raises:
        HTTPException: Code 404 lorsque le chunk n'existe pas.
    """
    with get_session() as session:
        deleted = ChunkService(session).delete(chunk_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chunk introuvable")
