"""Tests du router /chat, avec un ChatService factice injecté via dépendance FastAPI."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import chat as chat_router


class FakeChatService:
    """Implémentation factice de ChatService pour les tests du router."""

    async def ask(self, question, history, limit):
        """Fournit une réponse et une source fixes au endpoint testé.

        Args:
            question: Question envoyée par le client HTTP.
            history: Historique transmis par le client.
            limit: Nombre maximal de résultats demandé par la requête.

        Returns:
            dict: Réponse factice contenant le texte généré et une source.
        """
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
