"""Service métier pour la gestion des chunks (CRUD au-dessus du ChunkRepository)."""

from sqlalchemy.orm import Session

from database.chunk_repository import ChunkRepository
from database.models import Chunk
from logger import get_logger

logger = get_logger(__name__)


class ChunkService:
    """Opérations CRUD sur les chunks de documents."""

    def __init__(self, session: Session):
        self.session = session
        self.repository = ChunkRepository(session)

    def get_all(self) -> list[Chunk]:
        """Retourne tous les chunks."""
        return self.session.query(Chunk).all()

    def get_by_id(self, chunk_id: int) -> Chunk | None:
        """Retourne un chunk par son id, ou None si introuvable."""
        return self.repository.get_by_id(chunk_id)

    def count(self) -> int:
        """Retourne le nombre total de chunks."""
        return self.repository.count()

    def create(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        embedding_text: str,
        section: str | None = None,
        page: int | None = None,
        embedding: list[float] | None = None,
    ) -> Chunk:
        """Crée un nouveau chunk pour un document."""
        chunk = self.repository.create(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            embedding_text=embedding_text,
            section=section,
            page=page,
            embedding=embedding,
        )
        logger.info("Chunk %d créé pour le document %d", chunk.id, document_id)
        return chunk

    def update(self, chunk_id: int, fields: dict) -> Chunk | None:
        """Met à jour partiellement un chunk existant, ou None si introuvable."""
        chunk = self.repository.get_by_id(chunk_id)
        if not chunk:
            logger.warning("Chunk %d introuvable pour mise à jour", chunk_id)
            return None
        for field, value in fields.items():
            setattr(chunk, field, value)
        self.session.commit()
        self.session.refresh(chunk)
        logger.info("Chunk %d mis à jour", chunk_id)
        return chunk

    def delete(self, chunk_id: int) -> bool:
        """Supprime un chunk, retourne True si la suppression a eu lieu."""
        deleted = self.repository.delete(chunk_id)
        if deleted:
            logger.info("Chunk %d supprimé", chunk_id)
        else:
            logger.warning("Chunk %d introuvable pour suppression", chunk_id)
        return deleted
