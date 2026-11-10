"""Service métier pour la consultation des codes d'erreur (au-dessus de l'ErrorRepository)."""

from sqlalchemy.orm import Session

from database.error_repository import ErrorRepository
from database.models import ErrorCode
from logger import get_logger

logger = get_logger(__name__)


class ErrorService:
    """Expose les opérations de consultation des codes d'erreur."""

    def __init__(self, session: Session):
        """Prépare le service avec la session de travail de la requête.

        Args:
            session: Session SQLAlchemy transmise au dépôt de codes d'erreur.

        Returns:
            None: Le constructeur initialise le dépôt associé au service.
        """
        self.repository = ErrorRepository(session)

    def get_all(self) -> list[ErrorCode]:
        """Récupère tous les codes d'erreur, tous modèles confondus.

        Returns:
            list[ErrorCode]: Codes enregistrés, éventuellement une liste vide.
        """
        return self.repository.get_all()

    def get_by_model(self, model_id: int) -> list[ErrorCode]:
        """Récupère les codes d'erreur associés à un modèle.

        Args:
            model_id: Identifiant du modèle ciblé.

        Returns:
            list[ErrorCode]: Codes du modèle, éventuellement une liste vide.
        """
        return self.repository.get_error_codes_by_model(model_id)

    def get_all_unique_codes(self) -> list[str]:
        """Récupère les codes d'erreur distincts de tous les modèles.

        Returns:
            list[str]: Valeurs de codes uniques, triées par le dépôt.
        """
        return self.repository.get_all_unique_codes()
