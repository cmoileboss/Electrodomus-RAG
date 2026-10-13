"""Service d'ingestion : déclenche la conversion, le découpage et l'indexation des documents."""

from fastapi import BackgroundTasks

from doc_management.chunks_storing import process_all, process_single
from logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    """Orchestration de l'ingestion de documents, unitaire ou en masse."""

    def ingest_single(self, filepath: str) -> None:
        """Lance l'ingestion synchrone d'un fichier.

        Args:
            filepath: Chemin du document à convertir et indexer.

        Returns:
            None: Le traitement s'exécute jusqu'à son terme; ses erreurs sont propagées.
        """
        logger.info("Requête d'ingestion reçue pour '%s'", filepath)
        process_single(filepath)
        logger.info("Ingestion terminée pour '%s'", filepath)

    def ingest_all(self, background_tasks: BackgroundTasks) -> None:
        """Planifie l'ingestion globale dans le gestionnaire de tâches FastAPI.

        Args:
            background_tasks: Gestionnaire auquel ajouter le traitement ``process_all``.

        Returns:
            None: L'ingestion est ajoutée à la file des tâches d'arrière-plan.
        """
        logger.info("Ingestion globale démarrée en arrière-plan.")
        background_tasks.add_task(process_all)
