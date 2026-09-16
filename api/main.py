"""Point d'entrée de l'application FastAPI : création de l'app, cycle de vie et enregistrement des routers."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from database.database import init_db
from logger import get_logger
from api.routers import chat as chat_router
from api.routers import chunks as chunks_router
from api.routers import documents as documents_router
from api.routers import models as models_router

logger = get_logger(__name__)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise la base de données et charge le modèle d'embedding au démarrage."""
    init_db()
    logger.info("Initialisation de la base de données terminée.")
    logger.info("Chargement du modèle d'embedding : %s", EMBED_MODEL_ID)
    app.state.embed_model = await asyncio.to_thread(
        lambda: SentenceTransformer(EMBED_MODEL_ID, use_auth_token=HF_TOKEN)
    )
    logger.info("Modèle d'embedding chargé.")
    yield


app = FastAPI(title="Electrodomus RAG API", lifespan=lifespan)

app.include_router(chat_router.router)
app.include_router(chunks_router.router)
app.include_router(documents_router.router)
app.include_router(models_router.router)


@app.get("/health")
def health():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
