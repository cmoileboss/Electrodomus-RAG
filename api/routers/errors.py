"""Endpoint de lecture /error-codes pour lister les codes d'erreur (filtrable par modèle)."""

from fastapi import APIRouter
from pydantic import BaseModel

from api.services.error_service import ErrorService
from database.database import get_session
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/error-codes", tags=["error-codes"])


class ErrorCodeResponse(BaseModel):
    """Représentation publique d'un code d'erreur.

    Attributes:
        code: Valeur du code d'erreur.
        model_id: Identifiant du modèle concerné, absent pour la liste globale.
        signification: Explication du code, présente lorsque le filtre modèle est actif.
    """

    code: str
    model_id: int | None = None
    signification: str | None = None

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[ErrorCodeResponse])
def get_all(model_id: int | None = None):
    """Liste les codes d'erreur, éventuellement filtrés par modèle.

    Args:
        model_id: Identifiant du modèle à filtrer; ``None`` demande les codes
            uniques tous modèles confondus.

    Returns:
        list[ErrorCodeResponse]: Codes trouvés ou valeurs de codes distinctes.
    """
    with get_session() as session:
        service = ErrorService(session)
        if model_id is not None:
            codes = service.get_by_model(model_id)
            return [ErrorCodeResponse.model_validate(c) for c in codes]
        return [ErrorCodeResponse(code=code) for code in service.get_all_unique_codes()]
