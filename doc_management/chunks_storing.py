"""Points d'entrée d'ingestion : traitement d'un fichier unique ou de tous les documents du dossier source."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from database.database import get_session
from doc_management.doc_manager import DocManager
from logger import get_logger

logger = get_logger(__name__)

load_dotenv()

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 400))
DOC_FOLDER = Path("Documentation_Electrodomus")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html"}


def build_doc_manager() -> DocManager:
    """Construit un DocManager configuré à partir des variables d'environnement."""
    return DocManager(
        embed_model_id=EMBED_MODEL_ID,
        hf_token=HF_TOKEN,
        max_tokens=MAX_TOKENS,
    )


def process_single(filepath: str):
    """Ingère un unique fichier dans sa propre session de base de données."""
    logger.info("Traitement du fichier unique : %s", filepath)
    doc_manager = build_doc_manager()
    with get_session() as session:
        doc_manager.process_document(doc_source=filepath, session=session)
    logger.info("Traitement du fichier unique terminé : %s", filepath)


def _process_file(filepath: str, doc_manager: DocManager):
    """Traite un fichier dans son propre thread avec sa propre session."""
    with get_session() as session:
        doc_manager.process_document(doc_source=filepath, session=session)


def process_all():
    """Ingère en parallèle tous les fichiers supportés du dossier de documentation."""
    doc_manager = build_doc_manager()
    files = [
        str(f) for f in DOC_FOLDER.rglob("*")
        if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
    ]
    if not files:
        logger.warning("Aucun fichier support\u00e9 trouv\u00e9 dans %s", DOC_FOLDER)
        return
    logger.info("Ingestion de %d fichier(s) depuis %s", len(files), DOC_FOLDER)
    max_workers = min(4, len(files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_file, f, doc_manager): f for f in files}
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Erreur lors du traitement de %s : %s", filepath, e)
    logger.info("Ingestion globale termin\u00e9e (%d fichier(s) trait\u00e9s).", len(files))
