"""Accès aux données pour l'entité ErrorCode."""

from sqlalchemy.orm import Session

from database.models import ErrorCode
from logger import get_logger

logger = get_logger(__name__)


class ErrorRepository:
    """Accède aux codes d'erreur et à leurs associations en base de données."""

    def __init__(self, session: Session):
        """Initialise le dépôt avec une session SQLAlchemy existante.

        Args:
            session: Session utilisée pour les lectures et écritures en base.

        Returns:
            None: Le constructeur initialise le dépôt sans valeur de retour.
        """
        self.session = session

    def create(self, signification: str, client_behaviour: str, ass_intervention: str, code: str, model_id: int) -> ErrorCode:
        """Crée un code d'erreur pour un modèle et le persiste immédiatement.

        Args:
            signification: Explication du sens du code d'erreur.
            client_behaviour: Consignes à communiquer au client.
            ass_intervention: Intervention recommandée au technicien SAV.
            code: Valeur du code d'erreur.
            model_id: Identifiant du modèle concerné.

        Returns:
            ErrorCode: Code d'erreur persisté et actualisé depuis la base.
        """
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
        """Recherche un code d'erreur à partir de sa clé composite.

        Args:
            code: Valeur du code d'erreur.
            model_id: Identifiant du modèle auquel le code est rattaché.

        Returns:
            ErrorCode | None: Code trouvé, ou ``None`` si cette paire est absente.
        """
        return self.session.get(ErrorCode, (code, model_id))

    def get_by_code_and_model(self, code: str, model_id: int) -> ErrorCode | None:
        """Recherche un code d'erreur par code et modèle.

        Args:
            code: Valeur du code d'erreur.
            model_id: Identifiant du modèle auquel le code est rattaché.

        Returns:
            ErrorCode | None: Code trouvé, ou ``None`` si cette paire est absente.
        """
        return self.session.query(ErrorCode).filter_by(code=code, model_id=model_id).first()

    def get_all(self) -> list[ErrorCode]:
        """Récupère tous les codes d'erreur, tous modèles confondus.

        Returns:
            list[ErrorCode]: Codes enregistrés, éventuellement une liste vide.
        """
        return self.session.query(ErrorCode).all()

    def delete(self, code: str, model_id: int) -> bool:
        """Supprime le code d'erreur identifié par son code et son modèle.

        Args:
            code: Valeur du code d'erreur à supprimer.
            model_id: Identifiant du modèle concerné.

        Returns:
            bool: ``True`` si le code existait et a été supprimé, sinon ``False``.
        """
        error_code = self.get_by_id(code, model_id)
        if not error_code:
            logger.debug("ErrorCode '%s' (model %d) introuvable pour suppression", code, model_id)
            return False
        self.session.delete(error_code)
        self.session.commit()
        logger.debug("ErrorCode '%s' (model %d) supprimé", code, model_id)
        return True

    def delete_all(self) -> int:
        """Supprime tous les codes d'erreur enregistrés.

        Returns:
            int: Nombre de codes d'erreur supprimés.
        """
        error_codes = self.get_all()
        for error_code in error_codes:
            self.session.delete(error_code)
        self.session.commit()
        logger.debug("%d error_code(s) supprimé(s)", len(error_codes))
        return len(error_codes)

    def get_error_codes_by_model(self, model_id: int) -> list[ErrorCode]:
        """Récupère les codes d'erreur d'un modèle, triés par code croissant.

        Args:
            model_id: Identifiant du modèle ciblé.

        Returns:
            list[ErrorCode]: Codes associés au modèle, éventuellement une liste vide.
        """
        return self.session.query(ErrorCode).filter_by(model_id=model_id).order_by(ErrorCode.code).all()

    def get_all_unique_codes(self) -> list[str]:
        """Récupère les codes distincts, tous modèles confondus, par ordre croissant.

        Returns:
            list[str]: Valeurs de codes d'erreur sans doublons.
        """
        return [code for (code,) in self.session.query(ErrorCode.code).distinct().order_by(ErrorCode.code).all()]

    def get_error_codes_by_chunk(self, chunk_id: int) -> list[ErrorCode]:
        """Récupère les codes associés à un chunk.

        Args:
            chunk_id: Identifiant du chunk ciblé.

        Returns:
            list[ErrorCode]: Codes associés au chunk, éventuellement une liste vide.
        """
        return (
            self.session.query(ErrorCode)
            .join(ErrorCode.chunks)
            .filter_by(id=chunk_id)
            .all()
        )