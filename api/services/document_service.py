from datetime import datetime

from sqlalchemy.orm import Session

from database.document_repository import DocumentRepository
from database.models import Document
from logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, session: Session):
        self.session = session
        self.repository = DocumentRepository(session)

    def get_all(self) -> list[Document]:
        return self.repository.get_all()

    def get_by_id(self, document_id: int) -> Document | None:
        return self.repository.get_by_id(document_id)

    def create(self, title: str, filepath: str, date: datetime | None = None) -> Document:
        document = self.repository.create(title=title, filepath=filepath, date=date)
        logger.info("Document %d créé : '%s'", document.id, title)
        return document

    def update(self, document_id: int, fields: dict) -> Document | None:
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
        deleted = self.repository.delete(document_id)
        if deleted:
            logger.info("Document %d supprimé", document_id)
        else:
            logger.warning("Document %d introuvable pour suppression", document_id)
        return deleted
