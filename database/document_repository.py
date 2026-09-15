from sqlalchemy.orm import Session

from database.models import Document
from logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, title: str, filepath: str, date=None) -> Document:
        document = Document(title=title, filepath=filepath, date=date)
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        logger.debug("Document %d créé : '%s'", document.id, title)
        return document

    def get_by_id(self, document_id: int) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_title(self, title: str) -> Document | None:
        return self.session.query(Document).filter_by(title=title).first()

    def get_by_filepath(self, filepath: str) -> Document | None:
        return self.session.query(Document).filter_by(filepath=filepath).first()

    def get_all(self) -> list[Document]:
        return self.session.query(Document).all()

    def delete(self, document_id: int) -> bool:
        document = self.get_by_id(document_id)
        if not document:
            logger.debug("Document %d introuvable pour suppression", document_id)
            return False
        self.session.delete(document)
        self.session.commit()
        logger.debug("Document %d supprimé", document_id)
        return True
