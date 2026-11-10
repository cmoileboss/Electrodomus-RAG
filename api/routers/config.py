"""Endpoint /config : valeurs par défaut et choix possibles des réglages du pipeline RAG."""

from fastapi import APIRouter

from api.services.rag_config import ALLOWED_MODELS, MAX_CHUNK_LIMIT, RagConfig

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/")
def get_config():
    """Retourne les réglages par défaut du RAG et leurs bornes.

    Returns:
        dict: ``defaults`` (valeurs par défaut du serveur), ``models`` (modèles autorisés)
            et ``max_limit`` (valeur maximale d'un nombre de chunks).
    """
    return {
        "defaults": RagConfig().model_dump(),
        "models": ALLOWED_MODELS,
        "max_limit": MAX_CHUNK_LIMIT,
    }
