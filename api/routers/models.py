"""Endpoints CRUD /models pour gérer les modèles d'appareils Electrodomus."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.model_service import ModelService
from database.database import get_session
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class ModelCreate(BaseModel):
    """Données requises pour créer un modèle d'appareil.

    Attributes:
        name: Référence unique du modèle.
        type: Catégorie de l'appareil.
    """

    name: str
    type: str


class ModelUpdate(BaseModel):
    """Champs modifiables d'un modèle d'appareil.

    Attributes:
        name: Nouvelle référence du modèle, si elle est modifiée.
        type: Nouvelle catégorie de l'appareil, si elle est modifiée.
    """

    name: str | None = None
    type: str | None = None


class ModelResponse(BaseModel):
    """Représentation publique d'un modèle d'appareil.

    Attributes:
        id: Identifiant attribué en base.
        name: Référence unique du modèle.
        type: Catégorie de l'appareil.
    """

    id: int
    name: str
    type: str

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[ModelResponse])
def get_all():
    """Liste les modèles d'appareils enregistrés.

    Returns:
        list[ModelResponse]: Représentations API de tous les modèles.
    """
    with get_session() as session:
        models = ModelService(session).get_all()
        return [ModelResponse.model_validate(m) for m in models]


@router.get("/{model_id}", response_model=ModelResponse)
def get_by_id(model_id: int):
    """Récupère un modèle par son identifiant.

    Args:
        model_id: Identifiant du modèle demandé.

    Returns:
        ModelResponse: Modèle trouvé.

    Raises:
        HTTPException: Code 404 lorsque le modèle n'existe pas.
    """
    with get_session() as session:
        model = ModelService(session).get_by_id(model_id)
        if not model:
            logger.warning("Model introuvable : %s", model_id)
            raise HTTPException(status_code=404, detail="Model introuvable")
        return ModelResponse.model_validate(model)


@router.post("/", response_model=ModelResponse, status_code=201)
def create(body: ModelCreate):
    """Crée un modèle d'appareil.

    Args:
        body: Référence et catégorie du modèle à créer.

    Returns:
        ModelResponse: Modèle créé, avec son identifiant.
    """
    with get_session() as session:
        model = ModelService(session).create(name=body.name, type=body.type)
        return ModelResponse.model_validate(model)


@router.put("/{model_id}", response_model=ModelResponse)
def update_by_id(model_id: int, body: ModelUpdate):
    """Modifie les champs fournis d'un modèle existant.

    Args:
        model_id: Identifiant du modèle à modifier.
        body: Champs à mettre à jour; seuls ceux envoyés sont appliqués.

    Returns:
        ModelResponse: Modèle actualisé.

    Raises:
        HTTPException: Code 404 lorsque le modèle n'existe pas.
    """
    with get_session() as session:
        model = ModelService(session).update(model_id, body.model_dump(exclude_unset=True))
        if not model:
            raise HTTPException(status_code=404, detail="Model introuvable")
        return ModelResponse.model_validate(model)


@router.delete("/{model_id}", status_code=204)
def delete(model_id: int):
    """Supprime un modèle d'appareil.

    Args:
        model_id: Identifiant du modèle à supprimer.

    Returns:
        None: Réponse HTTP sans contenu (204) en cas de suppression.

    Raises:
        HTTPException: Code 404 lorsque le modèle n'existe pas.
    """
    with get_session() as session:
        deleted = ModelService(session).delete(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model introuvable")
