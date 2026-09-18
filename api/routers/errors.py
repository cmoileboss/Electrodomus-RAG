"""Endpoint de lecture /error-codes pour lister les codes d'erreur (filtrable par modèle)."""

from fastapi import APIRouter
from pydantic import BaseModel

from api.services.error_service import ErrorService
from database.database import get_session
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/error-codes", tags=["error-codes"])


class ErrorCodeResponse(BaseModel):
    """Représentation d'un code d'erreur retournée par l'API."""

    code: str
    model_id: int | None = None
    signification: str | None = None

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[ErrorCodeResponse])
def get_all(model_id: int | None = None):
    """Liste les codes d'erreur : filtrés par modèle si `model_id` est fourni, sinon uniques tous modèles confondus."""
    with get_session() as session:
        service = ErrorService(session)
        if model_id is not None:
            codes = service.get_by_model(model_id)
            return [ErrorCodeResponse.model_validate(c) for c in codes]
        return [ErrorCodeResponse(code=code) for code in service.get_all_unique_codes()]
