from sqlalchemy.orm import Session

from database.model_repository import ModelRepository
from database.models import Model
from logger import get_logger

logger = get_logger(__name__)


class ModelService:
    def __init__(self, session: Session):
        self.session = session
        self.repository = ModelRepository(session)

    def get_all(self) -> list[Model]:
        return self.repository.get_all()

    def get_by_id(self, model_id: int) -> Model | None:
        return self.repository.get_by_id(model_id)

    def create(self, name: str, type: str) -> Model:
        model = self.repository.create(name=name, type=type)
        logger.info("Model %d créé : '%s'", model.id, name)
        return model

    def update(self, model_id: int, fields: dict) -> Model | None:
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
        deleted = self.repository.delete(model_id)
        if deleted:
            logger.info("Model %d supprimé", model_id)
        else:
            logger.warning("Model %d introuvable pour suppression", model_id)
        return deleted
