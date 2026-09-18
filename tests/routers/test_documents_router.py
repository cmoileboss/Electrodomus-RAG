"""Tests du router /documents, avec DocumentService, IngestionService et get_session mockés."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import documents as documents_router


@pytest.fixture
def client():
    """Client de test FastAPI avec le router /documents."""
    app = FastAPI()
    app.include_router(documents_router.router)
    return TestClient(app)


def _fake_document(**overrides):
    """Construit un document factice avec des valeurs par défaut surchargeables."""
    defaults = dict(id=1, title="Manuel_FR-600", filepath="/docs/Manuel_FR-600.pdf", date=datetime(2026, 1, 1))
    defaults.update(overrides)
    return MagicMock(**defaults)


def test_get_all_returns_documents(client):
    """GET /documents/ retourne la liste des documents."""
    with patch.object(documents_router, "get_session", return_value=MagicMock()), \
         patch.object(documents_router, "DocumentService") as MockService:
        MockService.return_value.get_all.return_value = [_fake_document()]
        response = client.get("/documents/")

    assert response.status_code == 200
    assert response.json()[0]["title"] == "Manuel_FR-600"


def test_get_by_id_not_found_returns_404(client):
    """GET /documents/{id} retourne 404 quand le document est introuvable."""
    with patch.object(documents_router, "get_session", return_value=MagicMock()), \
         patch.object(documents_router, "DocumentService") as MockService:
        MockService.return_value.get_by_id.return_value = None
        response = client.get("/documents/999")

    assert response.status_code == 404


def test_create_document_returns_201(client):
    """POST /documents/ crée un document et retourne 201."""
    with patch.object(documents_router, "get_session", return_value=MagicMock()), \
         patch.object(documents_router, "DocumentService") as MockService:
        MockService.return_value.create.return_value = _fake_document()
        response = client.post("/documents/", json={"title": "Manuel_FR-600", "filepath": "/docs/x.pdf"})

    assert response.status_code == 201


def test_update_not_found_returns_404(client):
    """PUT /documents/{id} retourne 404 quand le document est introuvable."""
    with patch.object(documents_router, "get_session", return_value=MagicMock()), \
         patch.object(documents_router, "DocumentService") as MockService:
        MockService.return_value.update.return_value = None
        response = client.put("/documents/1", json={"title": "Nouveau titre"})

    assert response.status_code == 404


def test_delete_returns_204_when_found(client):
    """DELETE /documents/{id} supprime le document et retourne 204."""
    with patch.object(documents_router, "get_session", return_value=MagicMock()), \
         patch.object(documents_router, "DocumentService") as MockService:
        MockService.return_value.delete.return_value = True
        response = client.delete("/documents/1")

    assert response.status_code == 204


def test_delete_returns_404_when_not_found(client):
    """DELETE /documents/{id} retourne 404 quand le document est introuvable."""
    with patch.object(documents_router, "get_session", return_value=MagicMock()), \
         patch.object(documents_router, "DocumentService") as MockService:
        MockService.return_value.delete.return_value = False
        response = client.delete("/documents/1")

    assert response.status_code == 404


def test_ingest_single_success(client):
    """POST /documents/ingest lance l'ingestion et retourne 200."""
    with patch.object(documents_router, "IngestionService") as MockIngestion:
        response = client.post("/documents/ingest", json={"filepath": "doc.pdf"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "filepath": "doc.pdf"}
    MockIngestion.return_value.ingest_single.assert_called_once_with("doc.pdf")


def test_ingest_single_file_not_found_returns_404(client):
    """POST /documents/ingest retourne 404 si le fichier n'existe pas."""
    with patch.object(documents_router, "IngestionService") as MockIngestion:
        MockIngestion.return_value.ingest_single.side_effect = FileNotFoundError
        response = client.post("/documents/ingest", json={"filepath": "missing.pdf"})

    assert response.status_code == 404


def test_ingest_all_schedules_background_task(client):
    """POST /documents/ingest-all planifie l'ingestion en arrière-plan."""
    with patch.object(documents_router, "IngestionService") as MockIngestion:
        response = client.post("/documents/ingest-all")

    assert response.status_code == 200
    MockIngestion.return_value.ingest_all.assert_called_once()
