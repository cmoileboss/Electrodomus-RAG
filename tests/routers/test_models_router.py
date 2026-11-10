"""Tests du router /models, avec ModelService et get_session mockés."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import models as models_router


@pytest.fixture
def client():
    """Crée un client FastAPI avec le router /models.

    Returns:
        TestClient: Client configuré pour envoyer des requêtes au router de test.
    """
    app = FastAPI()
    app.include_router(models_router.router)
    return TestClient(app)


def _fake_model(**overrides):
    """Construit un modèle factice et remplace les valeurs demandées.

    Args:
        **overrides: Attributs à remplacer dans les valeurs par défaut du modèle.

    Returns:
        MagicMock: Objet factice exposant les attributs d'un modèle API.
    """
    defaults = dict(id=1, name="FR-600", type="four encastrable")
    defaults.update(overrides)
    # "name" est un paramètre spécial du constructeur MagicMock, il faut l'assigner après coup
    mock = MagicMock()
    for attr, value in defaults.items():
        setattr(mock, attr, value)
    return mock


def test_get_all_returns_models(client):
    """GET /models/ retourne la liste des modèles.

    Args:
        client: Client FastAPI fourni par la fixture de ce module.
    """
    with patch.object(models_router, "get_session", return_value=MagicMock()), \
         patch.object(models_router, "ModelService") as MockService:
        MockService.return_value.get_all.return_value = [_fake_model()]
        response = client.get("/models/")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "FR-600"


def test_get_by_id_not_found_returns_404(client):
    """GET /models/{id} retourne 404 quand le modèle est introuvable.

    Args:
        client: Client FastAPI fourni par la fixture de ce module.
    """
    with patch.object(models_router, "get_session", return_value=MagicMock()), \
         patch.object(models_router, "ModelService") as MockService:
        MockService.return_value.get_by_id.return_value = None
        response = client.get("/models/999")

    assert response.status_code == 404


def test_create_model_returns_201(client):
    """POST /models/ crée un modèle et retourne 201.

    Args:
        client: Client FastAPI fourni par la fixture de ce module.
    """
    with patch.object(models_router, "get_session", return_value=MagicMock()), \
         patch.object(models_router, "ModelService") as MockService:
        MockService.return_value.create.return_value = _fake_model()
        response = client.post("/models/", json={"name": "FR-600", "type": "four encastrable"})

    assert response.status_code == 201


def test_update_not_found_returns_404(client):
    """PUT /models/{id} retourne 404 quand le modèle est introuvable.

    Args:
        client: Client FastAPI fourni par la fixture de ce module.
    """
    with patch.object(models_router, "get_session", return_value=MagicMock()), \
         patch.object(models_router, "ModelService") as MockService:
        MockService.return_value.update.return_value = None
        response = client.put("/models/1", json={"name": "FR-650"})

    assert response.status_code == 404


def test_delete_returns_204_when_found(client):
    """DELETE /models/{id} supprime le modèle et retourne 204.

    Args:
        client: Client FastAPI fourni par la fixture de ce module.
    """
    with patch.object(models_router, "get_session", return_value=MagicMock()), \
         patch.object(models_router, "ModelService") as MockService:
        MockService.return_value.delete.return_value = True
        response = client.delete("/models/1")

    assert response.status_code == 204


def test_delete_returns_404_when_not_found(client):
    """DELETE /models/{id} retourne 404 quand le modèle est introuvable.

    Args:
        client: Client FastAPI fourni par la fixture de ce module.
    """
    with patch.object(models_router, "get_session", return_value=MagicMock()), \
         patch.object(models_router, "ModelService") as MockService:
        MockService.return_value.delete.return_value = False
        response = client.delete("/models/1")

    assert response.status_code == 404
