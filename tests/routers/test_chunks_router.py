"""Tests du router /chunks, avec ChunkService et get_session mockés."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import chunks as chunks_router


@pytest.fixture
def client():
    """Client de test FastAPI avec le router /chunks."""
    app = FastAPI()
    app.include_router(chunks_router.router)
    return TestClient(app)


def _fake_chunk(**overrides):
    """Construit un chunk factice avec des valeurs par défaut surchargeables."""
    defaults = dict(
        id=1, document_id=2, chunk_index=0, content="Contenu",
        embedding_text="Contenu contextualisé", section="Objet", page=1,
        created_at=datetime(2026, 1, 1),
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def test_get_all_returns_chunks(client):
    """GET /chunks/ retourne la liste des chunks."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.get_all.return_value = [_fake_chunk()]
        response = client.get("/chunks/")

    assert response.status_code == 200
    assert response.json()[0]["id"] == 1


def test_count_returns_repository_count(client):
    """GET /chunks/count retourne le nombre total de chunks."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.count.return_value = 42
        response = client.get("/chunks/count")

    assert response.status_code == 200
    assert response.json() == {"count": 42}


def test_get_by_id_found(client):
    """GET /chunks/{id} retourne le chunk quand il existe."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.get_by_id.return_value = _fake_chunk()
        response = client.get("/chunks/1")

    assert response.status_code == 200
    assert response.json()["content"] == "Contenu"


def test_get_by_id_not_found_returns_404(client):
    """GET /chunks/{id} retourne 404 quand le chunk est introuvable."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.get_by_id.return_value = None
        response = client.get("/chunks/999")

    assert response.status_code == 404


def test_create_chunk_returns_201(client):
    """POST /chunks/ crée un chunk et retourne 201."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.create.return_value = _fake_chunk()
        response = client.post("/chunks/", json={
            "document_id": 2, "chunk_index": 0, "content": "Contenu",
            "embedding_text": "Contenu contextualisé",
        })

    assert response.status_code == 201


def test_update_not_found_returns_404(client):
    """PUT /chunks/{id} retourne 404 quand le chunk est introuvable."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.update.return_value = None
        response = client.put("/chunks/1", json={"content": "Nouveau"})

    assert response.status_code == 404


def test_update_found_returns_200(client):
    """PUT /chunks/{id} met à jour le chunk et retourne 200."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.update.return_value = _fake_chunk(content="Nouveau")
        response = client.put("/chunks/1", json={"content": "Nouveau"})

    assert response.status_code == 200
    assert response.json()["content"] == "Nouveau"


def test_delete_not_found_returns_404(client):
    """DELETE /chunks/{id} retourne 404 quand le chunk est introuvable."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.delete.return_value = False
        response = client.delete("/chunks/1")

    assert response.status_code == 404


def test_delete_found_returns_204(client):
    """DELETE /chunks/{id} supprime le chunk et retourne 204."""
    with patch.object(chunks_router, "get_session", return_value=MagicMock()), \
         patch.object(chunks_router, "ChunkService") as MockService:
        MockService.return_value.delete.return_value = True
        response = client.delete("/chunks/1")

    assert response.status_code == 204
