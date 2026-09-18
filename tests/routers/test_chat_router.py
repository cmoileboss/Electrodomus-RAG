"""Tests du router /chat, avec un ChatService factice injecté via dépendance FastAPI."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import chat as chat_router


class FakeChatService:
    """Implémentation factice de ChatService pour les tests du router."""

    async def ask(self, question, history, limit):
        return {"answer": "Bonjour, voici la réponse.", "sources": [{"section": "Objet", "page": 1, "document_id": 1}]}

@pytest.fixture
def client():
    """Client de test FastAPI avec le router /chat et un ChatService factice."""
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[chat_router.get_chat_service] = lambda: FakeChatService()
    return TestClient(app)


def test_chat_returns_answer_and_updated_history(client):
    """POST /chat retourne la réponse, les sources et l'historique mis à jour."""
    response = client.post("/chat", json={"question": "Salut", "history": [], "limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Bonjour, voici la réponse."
    assert body["history"][-2] == {"role": "user", "content": "Salut"}
    assert body["history"][-1] == {"role": "assistant", "content": "Bonjour, voici la réponse."}
    assert body["sources"] == [{"section": "Objet", "page": 1, "document_id": 1}]
