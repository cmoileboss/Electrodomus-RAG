"""Service métier pour la consultation des codes d'erreur (au-dessus de l'ErrorRepository)."""

from sqlalchemy.orm import Session

from database.error_repository import ErrorRepository
from database.models import ErrorCode
from logger import get_logger

logger = get_logger(__name__)


class ErrorService:
    """Opérations de lecture sur les codes d'erreur."""

    def __init__(self, session: Session):
        self.repository = ErrorRepository(session)

    def get_all(self) -> list[ErrorCode]:
        """Retourne tous les codes d'erreur."""
        return self.repository.get_all()

    def get_by_model(self, model_id: int) -> list[ErrorCode]:
        """Retourne les codes d'erreur associés à un modèle donné."""
        return self.repository.get_error_codes_by_model(model_id)

    def get_all_unique_codes(self) -> list[str]:
        """Retourne les codes d'erreur uniques, tous modèles confondus."""
        return self.repository.get_all_unique_codes()
