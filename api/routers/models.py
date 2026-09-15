from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.model_service import ModelService
from database.database import get_session
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
        models = ModelService(session).get_all()
        return [ModelResponse.model_validate(m) for m in models]


@router.get("/{model_id}", response_model=ModelResponse)
def get_by_id(model_id: int):
    with get_session() as session:
        model = ModelService(session).get_by_id(model_id)
        if not model:
            logger.warning("Model introuvable : %s", model_id)
            raise HTTPException(status_code=404, detail="Model introuvable")
        return ModelResponse.model_validate(model)


@router.post("/", response_model=ModelResponse, status_code=201)
def create(body: ModelCreate):
    with get_session() as session:
        model = ModelService(session).create(name=body.name, type=body.type)
        return ModelResponse.model_validate(model)


@router.put("/{model_id}", response_model=ModelResponse)
def update_by_id(model_id: int, body: ModelUpdate):
    with get_session() as session:
        model = ModelService(session).update(model_id, body.model_dump(exclude_unset=True))
        if not model:
            raise HTTPException(status_code=404, detail="Model introuvable")
        return ModelResponse.model_validate(model)


@router.delete("/{model_id}", status_code=204)
def delete(model_id: int):
    with get_session() as session:
        deleted = ModelService(session).delete(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model introuvable")
