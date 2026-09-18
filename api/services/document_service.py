"""Service métier pour la gestion des documents (CRUD au-dessus du DocumentRepository)."""

from datetime import datetime

from sqlalchemy.orm import Session

from database.document_repository import DocumentRepository
from database.models import Document
from logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Opérations CRUD sur les documents."""

    def __init__(self, session: Session):
        self.session = session
        self.repository = DocumentRepository(session)

    def get_all(self) -> list[Document]:
        """Retourne tous les documents."""
        return self.repository.get_all()

    def get_by_id(self, document_id: int) -> Document | None:
        """Retourne un document par son id, ou None si introuvable."""
        return self.repository.get_by_id(document_id)

    def create(self, title: str, filepath: str, date: datetime | None = None) -> Document:
        """Crée un nouveau document."""
        document = self.repository.create(title=title, filepath=filepath, date=date)
        logger.info("Document %d créé : '%s'", document.id, title)
        return document

    def update(self, document_id: int, fields: dict) -> Document | None:
        """Met à jour partiellement un document existant, ou None si introuvable."""
        document = self.repository.get_by_id(document_id)
        if not document:
            logger.warning("Document %d introuvable pour mise à jour", document_id)
            return None
        for field, value in fields.items():
            setattr(document, field, value)
        self.session.commit()
        self.session.refresh(document)
        logger.info("Document %d mis à jour", document_id)
        return document

    def delete(self, document_id: int) -> bool:
        """Supprime un document, retourne True si la suppression a eu lieu."""
        deleted = self.repository.delete(document_id)
        if deleted:
            logger.info("Document %d supprimé", document_id)
        else:
            logger.warning("Document %d introuvable pour suppression", document_id)
        return deleted

    def delete_all(self) -> int:
        """Supprime tous les documents, retourne le nombre supprimé."""
        count = self.repository.delete_all()
        logger.info("%d document(s) supprimé(s)", count)
        return count
