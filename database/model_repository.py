from sqlalchemy.orm import Session

from database.models import Chunk, Model
from logger import get_logger

logger = get_logger(__name__)


class ModelRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str, type: str) -> Model:
        model = Model(name=name, type=type)
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)
        logger.debug("Model %d créé : '%s'", model.id, name)
        return model

    def get_by_id(self, model_id: int) -> Model | None:
        return self.session.get(Model, model_id)

    def get_by_name(self, name: str) -> Model | None:
        return self.session.query(Model).filter_by(name=name).first()

    def get_all(self) -> list[Model]:
        return self.session.query(Model).all()


    def link_chunk(self, model_id: int, chunk_id: int) -> bool:
        """Associe un chunk à un model."""
        model = self.get_by_id(model_id)
        chunk = self.session.get(Chunk, chunk_id)
        if not model or not chunk:
            logger.debug("Association impossible : model=%s, chunk=%s introuvable", model_id, chunk_id)
            return False
        if chunk not in model.chunks:
            model.chunks.append(chunk)
            self.session.commit()
            logger.debug("Chunk %d associé au model %d", chunk_id, model_id)
        return True

    def unlink_chunk(self, model_id: int, chunk_id: int) -> bool:
        """Dissocie un chunk d'un model."""
        model = self.get_by_id(model_id)
        chunk = self.session.get(Chunk, chunk_id)
        if not model or not chunk:
            logger.debug("Dissociation impossible : model=%s, chunk=%s introuvable", model_id, chunk_id)
            return False
        if chunk in model.chunks:
            model.chunks.remove(chunk)
            self.session.commit()
            logger.debug("Chunk %d dissocié du model %d", chunk_id, model_id)
        return True

    def get_chunks(self, model_id: int) -> list[Chunk]:
        model = self.get_by_id(model_id)
        return model.chunks if model else []

    def delete(self, model_id: int) -> bool:
        model = self.get_by_id(model_id)
        if not model:
            logger.debug("Model %d introuvable pour suppression", model_id)
            return False
        self.session.delete(model)
        self.session.commit()
        logger.debug("Model %d supprimé", model_id)
        return True
