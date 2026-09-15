from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.services.chat_service import ChatService


class FakeEmbedModel:
    def encode(self, text, normalize_embeddings=True):
        return np.array([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_ask_returns_answer_and_sources():
    fake_chunk = SimpleNamespace(id=1, section="Objet", page=2, document_id=5)
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.get_nearest.return_value = [fake_chunk]

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls), \
         patch("api.services.chat_service.build_rag_message", return_value="RAG CONTENT") as mock_build, \
         patch("api.services.chat_service.chat", return_value="La réponse") as mock_chat:
        service = ChatService(FakeEmbedModel())
        result = await service.ask("Ma question", [], limit=5)

    assert result["answer"] == "La réponse"
    assert result["sources"] == [{"section": "Objet", "page": 2, "document_id": 5}]
    mock_build.assert_called_once_with("Ma question", [fake_chunk])
    mock_chat.assert_called_once()


@pytest.mark.asyncio
async def test_ask_includes_history_in_messages_sent_to_chat():
    history = [SimpleNamespace(role="user", content="Bonjour")]
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.get_nearest.return_value = []

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls), \
         patch("api.services.chat_service.build_rag_message", return_value="RAG CONTENT"), \
         patch("api.services.chat_service.chat", return_value="ok") as mock_chat:
        service = ChatService(FakeEmbedModel())
        await service.ask("Question", history, limit=3)

    sent_messages = mock_chat.call_args.args[0]
    assert {"role": "user", "content": "Bonjour"} in sent_messages
    assert sent_messages[-1] == {"role": "user", "content": "RAG CONTENT"}


@pytest.mark.asyncio
async def test_stream_answer_yields_sources_tokens_then_done():
    fake_chunk = SimpleNamespace(id=1, section="Objet", page=2, document_id=5)
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.get_nearest.return_value = [fake_chunk]

    async def fake_stream_chat(messages):
        for token in ["Bon", "jour"]:
            yield token

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls), \
         patch("api.services.chat_service.build_rag_message", return_value="RAG CONTENT"), \
         patch("api.services.chat_service.stream_chat", fake_stream_chat):
        service = ChatService(FakeEmbedModel())
        events = [event async for event in service.stream_answer("Question", [], limit=5)]

    assert events[0] == {"type": "sources", "sources": [{"section": "Objet", "page": 2, "document_id": 5}]}
    assert [e["content"] for e in events if e["type"] == "token"] == ["Bon", "jour"]
    assert events[-1]["type"] == "done"
    assert events[-1]["history"][-1] == {"role": "assistant", "content": "Bonjour"}
