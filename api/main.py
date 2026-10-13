"""Point d'entrée de l'application FastAPI : création de l'app, cycle de vie et enregistrement des routers."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from database.database import init_db

from api.routers import chat as chat_router
from api.routers import chunks as chunks_router
from api.routers import config as config_router
from api.routers import documents as documents_router
from api.routers import errors as errors_router
from api.routers import models as models_router
from api.services.cross_encoder_reranker import Reranker

from logger import get_logger

logger = get_logger(__name__)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise les ressources partagées pendant le cycle de vie FastAPI.

    Args:
        app: Application dont l'état reçoit les modèles chargés au démarrage.

    Returns:
        AsyncIterator[None]: Contexte asynchrone qui suspend l'exécution pendant
            la durée de vie de l'application.
    """
    init_db()
    logger.info("Initialisation de la base de données terminée.")

    logger.info("Chargement du modèle d'embedding : %s", EMBED_MODEL_ID)
    app.state.embed_model = SentenceTransformer(EMBED_MODEL_ID, use_auth_token=HF_TOKEN)

    logger.info("Chargement du modèle de reranking : %s", CROSS_ENCODER_MODEL)
    app.state.reranker = Reranker(CROSS_ENCODER_MODEL)

    logger.info("Modèles chargés, API prête.")
    yield


app = FastAPI(title="Electrodomus RAG API", lifespan=lifespan)

app.include_router(chat_router.router)
app.include_router(chunks_router.router)
app.include_router(config_router.router)
app.include_router(documents_router.router)
app.include_router(errors_router.router)
app.include_router(models_router.router)


@app.get("/health")
def health():
    """Retourne l'état de disponibilité de l'API.

    Returns:
        dict[str, str]: Statut constant ``{"status": "ok"}``.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)