"""Accès aux données pour l'entité Document."""

from sqlalchemy.orm import Session

from database.models import Document
from logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """Opérations d'accès aux données pour l'entité Document."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, title: str, filepath: str, date=None) -> Document:
        """Crée et persiste un nouveau document."""
        document = Document(title=title, filepath=filepath, date=date)
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        logger.debug("Document %d créé : '%s'", document.id, title)
        return document

    def get_by_id(self, document_id: int) -> Document | None:
        """Retourne un document par son id, ou None si introuvable."""
        return self.session.get(Document, document_id)

    def get_by_title(self, title: str) -> Document | None:
        """Retourne un document par son titre, ou None si introuvable."""
        return self.session.query(Document).filter_by(title=title).first()

    def get_by_filepath(self, filepath: str) -> Document | None:
        """Retourne un document par son chemin de fichier, ou None si introuvable."""
        return self.session.query(Document).filter_by(filepath=filepath).first()

    def get_all(self) -> list[Document]:
        """Retourne tous les documents."""
        return self.session.query(Document).all()

    def delete(self, document_id: int) -> bool:
        """Supprime un document, retourne True si la suppression a eu lieu."""
        document = self.get_by_id(document_id)
        if not document:
            logger.debug("Document %d introuvable pour suppression", document_id)
            return False
        self.session.delete(document)
        self.session.commit()
        logger.debug("Document %d supprimé", document_id)
        return True

    def delete_all(self) -> int:
        """Supprime tous les documents (et leurs chunks associés), retourne le nombre supprimé."""
        documents = self.get_all()
        for document in documents:
            self.session.delete(document)
        self.session.commit()
        logger.debug("%d document(s) supprimé(s)", len(documents))
        return len(documents)
