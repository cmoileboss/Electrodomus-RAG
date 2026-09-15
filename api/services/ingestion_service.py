from fastapi import BackgroundTasks

from doc_management.chunks_storing import process_all, process_single
from logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    def ingest_single(self, filepath: str) -> None:
        """Convertit, découpe et indexe un document unique en base."""
        logger.info("Requête d'ingestion reçue pour '%s'", filepath)
        process_single(filepath)
        logger.info("Ingestion terminée pour '%s'", filepath)

    def ingest_all(self, background_tasks: BackgroundTasks) -> None:
        """Planifie l'ingestion de tous les documents en arrière-plan."""
        logger.info("Ingestion globale démarrée en arrière-plan.")
        background_tasks.add_task(process_all)
