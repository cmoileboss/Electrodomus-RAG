"""Service métier pour la gestion des modèles d'appareils (CRUD au-dessus du ModelRepository)."""

from sqlalchemy.orm import Session

from database.model_repository import ModelRepository
from database.models import Model
from logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """Applique la logique métier de gestion des modèles d'appareils."""

    def __init__(self, session: Session):
        """Prépare le service avec la session de travail de la requête.

        Args:
            session: Session SQLAlchemy partagée avec le dépôt de modèles.

        Returns:
            None: Le constructeur initialise le dépôt associé au service.
        """
        self.session = session
        self.repository = ModelRepository(session)

    def get_all(self) -> list[Model]:
        """Récupère tous les modèles enregistrés.

        Returns:
            list[Model]: Modèles présents en base, éventuellement une liste vide.
        """
        return self.repository.get_all()

    def get_by_id(self, model_id: int) -> Model | None:
        """Recherche un modèle par son identifiant.

        Args:
            model_id: Identifiant du modèle recherché.

        Returns:
            Model | None: Modèle trouvé, ou ``None`` si absent.
        """
        return self.repository.get_by_id(model_id)

    def create(self, name: str, type: str) -> Model:
        """Crée un modèle d'appareil et journalise sa création.

        Args:
            name: Référence unique du modèle d'appareil.
            type: Catégorie de l'appareil.

        Returns:
            Model: Modèle créé et persisté par le dépôt.
        """
        model = self.repository.create(name=name, type=type)
        logger.info("Model %d créé : '%s'", model.id, name)
        return model

    def update(self, model_id: int, fields: dict) -> Model | None:
        """Applique les champs fournis à un modèle existant.

        Args:
            model_id: Identifiant du modèle à modifier.
            fields: Noms et valeurs des attributs à mettre à jour.

        Returns:
            Model | None: Modèle actualisé, ou ``None`` si l'identifiant est absent.
        """
        model = self.repository.get_by_id(model_id)
        if not model:
            logger.warning("Model %d introuvable pour mise à jour", model_id)
            return None
        for field, value in fields.items():
            setattr(model, field, value)
        self.session.commit()
        self.session.refresh(model)
        logger.info("Model %d mis à jour", model_id)
        return model

    def delete(self, model_id: int) -> bool:
        """Supprime un modèle et journalise le résultat.

        Args:
            model_id: Identifiant du modèle à supprimer.

        Returns:
            bool: ``True`` si un modèle a été supprimé, sinon ``False``.
        """
        deleted = self.repository.delete(model_id)
        if deleted:
            logger.info("Model %d supprimé", model_id)
        else:
            logger.warning("Model %d introuvable pour suppression", model_id)
        return deleted
