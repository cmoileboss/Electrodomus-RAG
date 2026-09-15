import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import chat as chat_router


class FakeChatService:
    async def ask(self, question, history, limit):
        return {"answer": "Bonjour, voici la réponse.", "sources": [{"section": "Objet", "page": 1, "document_id": 1}]}

    async def stream_answer(self, question, history, limit):
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "content": "Bon"}
        yield {"type": "token", "content": "jour"}
        yield {"type": "done", "history": [{"role": "assistant", "content": "Bonjour"}]}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[chat_router.get_chat_service] = lambda: FakeChatService()
    return TestClient(app)


def test_chat_returns_answer_and_updated_history(client):
    response = client.post("/chat", json={"question": "Salut", "history": [], "limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Bonjour, voici la réponse."
    assert body["history"][-2] == {"role": "user", "content": "Salut"}
    assert body["history"][-1] == {"role": "assistant", "content": "Bonjour, voici la réponse."}
    assert body["sources"] == [{"section": "Objet", "page": 1, "document_id": 1}]


def test_chat_stream_returns_sse_events(client):
    response = client.post("/chat/stream", json={"question": "Salut"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "sources"' in response.text
    assert 'data: {"type": "token", "content": "Bon"}' in response.text
    assert 'data: {"type": "done"' in response.text
