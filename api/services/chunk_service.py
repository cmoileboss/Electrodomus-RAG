"""Service métier pour la gestion des chunks (CRUD au-dessus du ChunkRepository)."""

from sqlalchemy.orm import Session

from database.chunk_repository import ChunkRepository
from database.models import Chunk
from logger import get_logger

logger = get_logger(__name__)


class ChunkService:
    """Applique la logique métier de gestion des chunks de documents."""

    def __init__(self, session: Session):
        """Prépare le service avec la session de travail de la requête.

        Args:
            session: Session SQLAlchemy partagée avec le dépôt de chunks.

        Returns:
            None: Le constructeur initialise le dépôt associé au service.
        """
        self.session = session
        self.repository = ChunkRepository(session)

    def get_all(self) -> list[Chunk]:
        """Récupère tous les chunks enregistrés.

        Returns:
            list[Chunk]: Chunks présents en base, éventuellement une liste vide.
        """
        return self.session.query(Chunk).all()

    def get_by_id(self, chunk_id: int) -> Chunk | None:
        """Recherche un chunk par son identifiant.

        Args:
            chunk_id: Identifiant du chunk recherché.

        Returns:
            Chunk | None: Chunk trouvé, ou ``None`` si absent.
        """
        return self.repository.get_by_id(chunk_id)

    def count(self) -> int:
        """Compte les chunks enregistrés.

        Returns:
            int: Nombre total de chunks.
        """
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
        """Crée un chunk et consigne sa création.

        Args:
            document_id: Identifiant du document auquel le chunk appartient.
            chunk_index: Position du chunk dans le document.
            content: Texte extrait du document.
            embedding_text: Texte contextualisé utilisé pour l'embedding.
            section: Titre de la section source, s'il existe.
            page: Numéro de page source, s'il existe.
            embedding: Vecteur du chunk, facultatif.

        Returns:
            Chunk: Chunk créé et persisté par le dépôt.
        """
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
        """Applique les champs fournis à un chunk existant.

        Args:
            chunk_id: Identifiant du chunk à modifier.
            fields: Noms et valeurs des attributs à mettre à jour.

        Returns:
            Chunk | None: Chunk actualisé, ou ``None`` si l'identifiant est absent.
        """
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
        """Supprime un chunk et journalise le résultat.

        Args:
            chunk_id: Identifiant du chunk à supprimer.

        Returns:
            bool: ``True`` si un chunk a été supprimé, sinon ``False``.
        """
        deleted = self.repository.delete(chunk_id)
        if deleted:
            logger.info("Chunk %d supprimé", chunk_id)
        else:
            logger.warning("Chunk %d introuvable pour suppression", chunk_id)
        return deleted
