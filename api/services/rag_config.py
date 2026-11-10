"""Paramètres du pipeline RAG modifiables à chaque requête depuis le frontend."""

import os

from pydantic import BaseModel, Field, field_validator

# L'import du dépôt charge le fichier .env avant la lecture des variables d'environnement ci-dessous
from database.chunk_repository import CHUNK_LIMIT_BM25, CHUNK_LIMIT_VECTORIAL

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CHUNK_LIMIT_RRF = int(os.getenv("CHUNK_LIMIT_RRF", "10"))
TOP_K_RERANKER = int(os.getenv("TOP_K_RERANKER", "3"))

# Évite d'envoyer à Groq un nom de modèle arbitraire fourni par le client
ALLOWED_MODELS = [
    model
    for model in dict.fromkeys(["openai/gpt-oss-20b", "openai/gpt-oss-120b", DEFAULT_MODEL])
    if model
]
MAX_CHUNK_LIMIT = 200


class RagConfig(BaseModel):
    """Réglages du pipeline RAG; chaque champ non transmis prend sa valeur par défaut.

    Attributes:
        model: Modèle Groq utilisé pour la génération, parmi ``ALLOWED_MODELS``.
        vector_limit: Nombre de chunks retournés par la recherche vectorielle.
        bm25_limit: Nombre de chunks retournés par la recherche BM25.
        rrf_limit: Nombre de chunks conservés après la fusion RRF.
        rerank_top_k: Nombre de chunks conservés après le reranking, envoyés au LLM.
    """

    model: str = DEFAULT_MODEL
    vector_limit: int = Field(default=CHUNK_LIMIT_VECTORIAL, ge=1, le=MAX_CHUNK_LIMIT)
    bm25_limit: int = Field(default=CHUNK_LIMIT_BM25, ge=1, le=MAX_CHUNK_LIMIT)
    rrf_limit: int = Field(default=CHUNK_LIMIT_RRF, ge=1, le=MAX_CHUNK_LIMIT)
    rerank_top_k: int = Field(default=TOP_K_RERANKER, ge=1, le=MAX_CHUNK_LIMIT)

    @field_validator("model")
    @classmethod
    def _model_must_be_allowed(cls, value: str) -> str:
        """Refuse les modèles absents de la liste autorisée.

        Args:
            value: Nom de modèle fourni par le client.

        Returns:
            str: Le même nom, s'il est autorisé.

        Raises:
            ValueError: Si le modèle n'est pas dans ``ALLOWED_MODELS``.
        """
        if value not in ALLOWED_MODELS:
            raise ValueError(f"Modèle non autorisé : {value}. Valeurs possibles : {', '.join(ALLOWED_MODELS)}")
        return value
