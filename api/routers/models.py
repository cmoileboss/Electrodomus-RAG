from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.database import get_session
from database.model_repository import ModelRepository
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class ModelCreate(BaseModel):
    name: str
    type: str


class ModelUpdate(BaseModel):
    name: str | None = None
    type: str | None = None


class ModelResponse(BaseModel):
    id: int
    name: str
    type: str

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[ModelResponse])
def get_all():
    with get_session() as session:
        models = ModelRepository(session).get_all()
        return [ModelResponse.model_validate(m) for m in models]


@router.get("/{model_id}", response_model=ModelResponse)
def get_by_id(model_id: int):
    with get_session() as session:
        model = ModelRepository(session).get_by_id(model_id)
        if not model:
            logger.warning("Model introuvable : %s", model_id)
            raise HTTPException(status_code=404, detail="Model introuvable")
        return ModelResponse.model_validate(model)


@router.post("/", response_model=ModelResponse, status_code=201)
def create(body: ModelCreate):
    with get_session() as session:
        model = ModelRepository(session).create(name=body.name, type=body.type)
        logger.info("Model %d créé : '%s'", model.id, model.name)
        return ModelResponse.model_validate(model)


@router.put("/{model_id}", response_model=ModelResponse)
def update_by_id(model_id: int, body: ModelUpdate):
    with get_session() as session:
        model = ModelRepository(session).get_by_id(model_id)
        if not model:
            logger.warning("Model introuvable pour mise à jour : %s", model_id)
            raise HTTPException(status_code=404, detail="Model introuvable")
        for field, value in body.model_dump(exclude_unset=True).items():
            setattr(model, field, value)
        session.commit()
        session.refresh(model)
        logger.info("Model %d mis à jour", model_id)
        return ModelResponse.model_validate(model)


@router.delete("/{model_id}", status_code=204)
def delete(model_id: int):
    with get_session() as session:
        deleted = ModelRepository(session).delete(model_id)
    if not deleted:
        logger.warning("Model introuvable pour suppression : %s", model_id)
        raise HTTPException(status_code=404, detail="Model introuvable")
    logger.info("Model %d supprimé", model_id)
