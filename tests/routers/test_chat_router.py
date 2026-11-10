"""Tests du router /chat, avec un ChatService factice injecté via dépendance FastAPI."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import chat as chat_router


class FakeChatService:
    """Implémentation factice de ChatService pour les tests du router."""

    last_kwargs = None

    async def ask(self, question, history, **kwargs):
        """Fournit une réponse et une source fixes au endpoint testé.

        Args:
            question: Question envoyée par le client HTTP.
            history: Historique transmis par le client.
            **kwargs: Filtres et réglages transmis par le router, mémorisés pour les tests.

        Returns:
            dict: Réponse factice contenant le texte généré et une source.
        """
        FakeChatService.last_kwargs = kwargs
        return {"answer": "Bonjour, voici la réponse.", "sources": [{"section": "Objet", "page": 1, "document_id": 1}]}

@pytest.fixture
def client():
    """Crée un client FastAPI avec le router /chat et un service factice.

    Returns:
        TestClient: Client configuré pour envoyer des requêtes au router de test.
    """
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[chat_router.get_chat_service] = lambda: FakeChatService()
    return TestClient(app)


def test_chat_returns_answer_and_updated_history(client):
    """POST /chat retourne la réponse, les sources et l'historique mis à jour.

    Args:
        client: Client FastAPI fourni par la fixture de ce module.
    """
    response = client.post("/chat", json={"question": "Salut", "history": [], "limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Bonjour, voici la réponse."
    assert body["history"][-2] == {"role": "user", "content": "Salut"}
    assert body["history"][-1] == {"role": "assistant", "content": "Bonjour, voici la réponse."}
    assert body["sources"] == [{"section": "Objet", "page": 1, "document_id": 1}]


def test_chat_forwards_rag_config_to_service(client):
    """POST /chat transmet la config RAG du client au service de chat."""
    config = {"model": "openai/gpt-oss-120b", "vector_limit": 12, "bm25_limit": 13, "rrf_limit": 6, "rerank_top_k": 4}
    response = client.post("/chat", json={"question": "Salut", "config": config})

    assert response.status_code == 200
    assert FakeChatService.last_kwargs["config"].model_dump() == config


def test_chat_rejects_unknown_model(client):
    """POST /chat refuse un modèle LLM hors de la liste autorisée."""
    response = client.post("/chat", json={"question": "Salut", "config": {"model": "autre/modele"}})

    assert response.status_code == 422
