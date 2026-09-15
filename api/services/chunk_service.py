from sqlalchemy.orm import Session

from database.chunk_repository import ChunkRepository
from database.models import Chunk
from logger import get_logger

logger = get_logger(__name__)


class ChunkService:
    def __init__(self, session: Session):
        self.session = session
        self.repository = ChunkRepository(session)

    def get_all(self) -> list[Chunk]:
        return self.session.query(Chunk).all()

    def get_by_id(self, chunk_id: int) -> Chunk | None:
        return self.repository.get_by_id(chunk_id)

    def count(self) -> int:
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
        deleted = self.repository.delete(chunk_id)
        if deleted:
            logger.info("Chunk %d supprimé", chunk_id)
        else:
            logger.warning("Chunk %d introuvable pour suppression", chunk_id)
        return deleted
