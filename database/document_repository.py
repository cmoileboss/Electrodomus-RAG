"""Accès aux données pour l'entité Document."""

from sqlalchemy.orm import Session

from database.models import Document
from logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """Accède aux documents persistés et exécute leurs opérations CRUD."""

    def __init__(self, session: Session):
        """Initialise le dépôt avec une session SQLAlchemy existante.

        Args:
            session: Session utilisée pour les lectures et écritures en base.

        Returns:
            None: Le constructeur initialise le dépôt sans valeur de retour.
        """
        self.session = session

    def create(self, title: str, filepath: str, date=None) -> Document:
        """Crée un document et le persiste immédiatement.

        Args:
            title: Titre unique du document.
            filepath: Chemin du fichier source associé au document.
            date: Date du document, si elle est connue.

        Returns:
            Document: Document persisté et actualisé depuis la base.
        """
        document = Document(title=title, filepath=filepath, date=date)
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        logger.debug("Document %d créé : '%s'", document.id, title)
        return document

    def get_by_id(self, document_id: int) -> Document | None:
        """Recherche un document par son identifiant.

        Args:
            document_id: Identifiant du document recherché.

        Returns:
            Document | None: Document trouvé, ou ``None`` si absent.
        """
        return self.session.get(Document, document_id)

    def get_by_title(self, title: str) -> Document | None:
        """Recherche un document par son titre.

        Args:
            title: Titre exact du document recherché.

        Returns:
            Document | None: Premier document trouvé, ou ``None`` si absent.
        """
        return self.session.query(Document).filter_by(title=title).first()

    def get_by_filepath(self, filepath: str) -> Document | None:
        """Recherche un document par son chemin de fichier.

        Args:
            filepath: Chemin exact du fichier associé au document.

        Returns:
            Document | None: Premier document trouvé, ou ``None`` si absent.
        """
        return self.session.query(Document).filter_by(filepath=filepath).first()

    def get_all(self) -> list[Document]:
        """Récupère tous les documents enregistrés.

        Returns:
            list[Document]: Documents présents en base, éventuellement une liste vide.
        """
        return self.session.query(Document).all()

    def delete(self, document_id: int) -> bool:
        """Supprime un document par son identifiant.

        Args:
            document_id: Identifiant du document à supprimer.

        Returns:
            bool: ``True`` si le document existait et a été supprimé, sinon ``False``.
        """
        document = self.get_by_id(document_id)
        if not document:
            logger.debug("Document %d introuvable pour suppression", document_id)
            return False
        self.session.delete(document)
        self.session.commit()
        logger.debug("Document %d supprimé", document_id)
        return True

    def delete_all(self) -> int:
        """Supprime tous les documents et leurs chunks associés.

        Returns:
            int: Nombre de documents supprimés.
        """
        documents = self.get_all()
        for document in documents:
            self.session.delete(document)
        self.session.commit()
        logger.debug("%d document(s) supprimé(s)", len(documents))
        return len(documents)
