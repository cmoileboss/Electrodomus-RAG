"""Accès aux données pour l'entité ErrorCode."""

from sqlalchemy.orm import Session

from database.models import ErrorCode
from logger import get_logger

logger = get_logger(__name__)


class ErrorRepository:
    """Opérations d'accès aux données pour l'entité ErrorCode."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, signification: str, client_behaviour: str, ass_intervention: str, code: str, model_id: int) -> ErrorCode:
        """Crée et persiste un nouveau code d'erreur rattaché à un modèle."""
        error_code = ErrorCode(
            code=code,
            model_id=model_id,
            signification=signification,
            client_behaviour=client_behaviour,
            ass_intervention=ass_intervention,
        )
        self.session.add(error_code)
        self.session.commit()
        self.session.refresh(error_code)
        logger.debug("ErrorCode '%s' créé pour le model %d", code, model_id)
        return error_code

    def get_by_id(self, code: str, model_id: int) -> ErrorCode | None:
        """Retourne un code d'erreur par son code et son model_id, ou None si introuvable."""
        return self.session.get(ErrorCode, (code, model_id))

    def get_all(self) -> list[ErrorCode]:
        """Retourne tous les codes d'erreur."""
        return self.session.query(ErrorCode).all()

    def delete(self, code: str, model_id: int) -> bool:
        """Supprime un code d'erreur, retourne True si la suppression a eu lieu."""
        error_code = self.get_by_id(code, model_id)
        if not error_code:
            logger.debug("ErrorCode '%s' (model %d) introuvable pour suppression", code, model_id)
            return False
        self.session.delete(error_code)
        self.session.commit()
        logger.debug("ErrorCode '%s' (model %d) supprimé", code, model_id)
        return True

    def delete_all(self) -> int:
        """Supprime tous les codes d'erreur, retourne le nombre supprimé."""
        error_codes = self.get_all()
        for error_code in error_codes:
            self.session.delete(error_code)
        self.session.commit()
        logger.debug("%d error_code(s) supprimé(s)", len(error_codes))
        return len(error_codes)

    def get_error_codes_by_model(self, model_id: int) -> list[ErrorCode]:
        """Retourne tous les codes d'erreur associés à un modèle donné, triés par code croissant."""
        return self.session.query(ErrorCode).filter_by(model_id=model_id).order_by(ErrorCode.code).all()

    def get_all_unique_codes(self) -> list[str]:
        """Retourne les codes d'erreur uniques, tous modèles confondus, triés par ordre croissant."""
        return [code for (code,) in self.session.query(ErrorCode.code).distinct().order_by(ErrorCode.code).all()]

    def get_error_codes_by_chunk(self, chunk_id: int) -> list[ErrorCode]:
        """Retourne tous les codes d'erreur associés à un chunk donné."""
        return (
            self.session.query(ErrorCode)
            .join(ErrorCode.chunks)
            .filter_by(id=chunk_id)
            .all()
        )