"""Points d'entrée d'ingestion : traitement d'un fichier unique ou de tous les documents du dossier source."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from database.database import get_session
from database.document_repository import DocumentRepository
from doc_management.doc_manager import DocManager
from logger import get_logger

logger = get_logger(__name__)

load_dotenv()

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 400))
DOC_FOLDER = Path("Documentation_Electrodomus")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".xlsx", ".txt"}


def build_doc_manager() -> DocManager:
    """Construit un gestionnaire de documents avec la configuration d'environnement.

    Returns:
        DocManager: Gestionnaire configuré avec le modèle et la limite de tokens.
    """
    return DocManager(
        embed_model_id=EMBED_MODEL_ID,
        hf_token=HF_TOKEN,
        max_tokens=MAX_TOKENS,
    )


def process_single(filepath: str):
    """Ingère un fichier unique en ouvrant sa session de base de données.

    Args:
        filepath: Chemin du fichier source à traiter.

    Returns:
        None: Le traitement est effectué dans la session puis celle-ci est fermée.
    """
    logger.info("Traitement du fichier unique : %s", filepath)
    doc_manager = build_doc_manager()
    with get_session() as session:
        doc_manager.process_document(doc_source=filepath, session=session)
    logger.info("Traitement du fichier unique terminé : %s", filepath)


def _process_file(filepath: str, doc_manager: DocManager):
    """Traite un fichier dans une session dédiée, pour l'exécution concurrente.

    Args:
        filepath: Chemin du fichier source à traiter.
        doc_manager: Gestionnaire de documents partagé par les tâches d'ingestion.

    Returns:
        None: Le fichier est traité et les changements sont validés par son gestionnaire.
    """
    with get_session() as session:
        doc_manager.process_document(doc_source=filepath, session=session)


def process_all():
    """Découvre et ingère les nouveaux fichiers supportés en parallèle.

    Les fichiers dont le nom est déjà présent en base sont ignorés. Les erreurs
    de tâches individuelles sont consignées sans interrompre les autres traitements.

    Returns:
        None: Les résultats sont persistés par les tâches concurrentes.
    """
    discovered_files = [
        str(f) for f in DOC_FOLDER.rglob("*")
        if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
    ]
    if not discovered_files:
        logger.warning("Aucun fichier support\u00e9 trouv\u00e9 dans %s", DOC_FOLDER)
        return

    with get_session() as session:
        existing_filenames = {
            Path(document.filepath).name
            for document in DocumentRepository(session).get_all()
        }

    files = [
        filepath for filepath in discovered_files
        if Path(filepath).name not in existing_filenames
    ]
    skipped_count = len(discovered_files) - len(files)
    if skipped_count:
        logger.info("%d document(s) déjà présent(s) en base ignoré(s).", skipped_count)
    if not files:
        logger.info("Tous les documents supportés sont déjà présents en base.")
        return

    doc_manager = build_doc_manager()
    logger.info("Ingestion de %d fichier(s) depuis %s", len(files), DOC_FOLDER)
    max_workers = min(12, len(files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_file, f, doc_manager): f for f in files}
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Erreur lors du traitement de %s : %s", filepath, e)
    logger.info("Ingestion globale termin\u00e9e (%d fichier(s) trait\u00e9s).", len(files))
