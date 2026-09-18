"""Tests unitaires de IngestionService, avec les fonctions d'ingestion mockées."""

from unittest.mock import MagicMock, patch

import pytest

from api.services.ingestion_service import IngestionService


def test_ingest_single_calls_process_single():
    """ingest_single() appelle process_single() avec le chemin fourni."""
    with patch("api.services.ingestion_service.process_single") as mock_process:
        IngestionService().ingest_single("Documentation_Electrodomus/doc.pdf")

    mock_process.assert_called_once_with("Documentation_Electrodomus/doc.pdf")


def test_ingest_single_propagates_file_not_found():
    """ingest_single() propage FileNotFoundError si le fichier n'existe pas."""
    with patch("api.services.ingestion_service.process_single", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            IngestionService().ingest_single("missing.pdf")


def test_ingest_all_schedules_background_task():
    """ingest_all() planifie process_all() en tâche d'arrière-plan."""
    background_tasks = MagicMock()
    with patch("api.services.ingestion_service.process_all") as mock_process_all:
        IngestionService().ingest_all(background_tasks)

    background_tasks.add_task.assert_called_once_with(mock_process_all)
