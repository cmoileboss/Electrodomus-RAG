"""Accès aux données pour les modèles d'appareils et leurs associations avec les chunks."""

from sqlalchemy.orm import Session

from database.models import Chunk, ErrorCode, Model
from logger import get_logger

logger = get_logger(__name__)


class ModelRepository:
    """Accède aux modèles d'appareils et à leurs associations avec les chunks."""

    def __init__(self, session: Session):
        """Initialise le dépôt avec une session SQLAlchemy existante.

        Args:
            session: Session utilisée pour les lectures et écritures en base.

        Returns:
            None: Le constructeur initialise le dépôt sans valeur de retour.
        """
        self.session = session

    def create(self, name: str, type: str) -> Model:
        """Crée un modèle d'appareil et le persiste immédiatement.

        Args:
            name: Référence unique du modèle, par exemple ``FR-600``.
            type: Catégorie de l'appareil, par exemple ``four encastrable``.

        Returns:
            Model: Modèle persisté et actualisé depuis la base.
        """
        model = Model(name=name, type=type)
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)
        logger.debug("Model %d créé : '%s'", model.id, name)
        return model

    def get_by_id(self, model_id: int) -> Model | None:
        """Recherche un modèle par son identifiant.

        Args:
            model_id: Identifiant du modèle recherché.

        Returns:
            Model | None: Modèle trouvé, ou ``None`` si absent.
        """
        return self.session.get(Model, model_id)

    def get_by_name(self, name: str) -> Model | None:
        """Recherche un modèle par sa référence.

        Args:
            name: Référence exacte du modèle recherché.

        Returns:
            Model | None: Premier modèle correspondant, ou ``None`` si absent.
        """
        return self.session.query(Model).filter_by(name=name).first()

    def get_all(self) -> list[Model]:
        """Récupère tous les modèles enregistrés.

        Returns:
            list[Model]: Modèles présents en base, éventuellement une liste vide.
        """
        return self.session.query(Model).all()


    def link_chunk(self, model_id: int, chunk_id: int) -> bool:
        """Associe un chunk à un modèle si les deux existent.

        Args:
            model_id: Identifiant du modèle destinataire.
            chunk_id: Identifiant du chunk à associer.

        Returns:
            bool: ``True`` si les deux entités existent (y compris si l'association
                existait déjà), sinon ``False``.
        """
        model = self.get_by_id(model_id)
        chunk = self.session.get(Chunk, chunk_id)
        if not model or not chunk:
            logger.debug("Association impossible : model=%s, chunk=%s introuvable", model_id, chunk_id)
            return False
        if chunk not in model.chunks:
            model.chunks.append(chunk)
            self.session.commit()
            logger.debug("Chunk %d associé au model %d", chunk_id, model_id)
        return True

    def unlink_chunk(self, model_id: int, chunk_id: int) -> bool:
        """Retire l'association entre un chunk et un modèle.

        Args:
            model_id: Identifiant du modèle concerné.
            chunk_id: Identifiant du chunk à dissocier.

        Returns:
            bool: ``True`` si les deux entités existent (y compris sans association
                préalable), sinon ``False``.
        """
        model = self.get_by_id(model_id)
        chunk = self.session.get(Chunk, chunk_id)
        if not model or not chunk:
            logger.debug("Dissociation impossible : model=%s, chunk=%s introuvable", model_id, chunk_id)
            return False
        if chunk in model.chunks:
            model.chunks.remove(chunk)
            self.session.commit()
            logger.debug("Chunk %d dissocié du model %d", chunk_id, model_id)
        return True

    def get_chunks(self, model_id: int) -> list[Chunk]:
        """Récupère les chunks associés à un modèle.

        Args:
            model_id: Identifiant du modèle recherché.

        Returns:
            list[Chunk]: Chunks associés, ou une liste vide si le modèle est absent.
        """
        model = self.get_by_id(model_id)
        return model.chunks if model else []

    def delete(self, model_id: int) -> bool:
        """Supprime un modèle par son identifiant.

        Args:
            model_id: Identifiant du modèle à supprimer.

        Returns:
            bool: ``True`` si le modèle existait et a été supprimé, sinon ``False``.
        """
        model = self.get_by_id(model_id)
        if not model:
            logger.debug("Model %d introuvable pour suppression", model_id)
            return False
        self.session.delete(model)
        self.session.commit()
        logger.debug("Model %d supprimé", model_id)
        return True
