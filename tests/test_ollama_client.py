from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ollama_client import build_rag_message, chat


def test_build_rag_message_with_chunks():
    chunks = [
        SimpleNamespace(id=1, document_id=5, section="Objet", page=2, content="Contenu A"),
        SimpleNamespace(id=2, document_id=None, section=None, page=None, content="Contenu B"),
    ]
    result = build_rag_message("Ma question ?", chunks)
    assert "Contenu A" in result
    assert "Source : chunk 1, document 5, Objet, page 2" in result
    assert "Source : chunk 2, document ?, inconnue, page ?" in result
    assert "QUESTION : Ma question ?" in result


def test_build_rag_message_without_chunks():
    result = build_rag_message("Ma question ?", [])
    assert "Aucun extrait pertinent trouvé." in result
    assert "QUESTION : Ma question ?" in result


def test_chat_non_streaming_returns_message_content():
    fake_response = SimpleNamespace(
        ok=True, json=lambda: {"message": {"content": "Bonjour"}}
    )
    with patch("ollama_client.requests.post", return_value=fake_response) as mock_post:
        result = chat([{"role": "user", "content": "Salut"}], stream=False)
    assert result == "Bonjour"
    mock_post.assert_called_once()


def test_chat_raises_on_error_response():
    fake_response = SimpleNamespace(ok=False, status_code=500, text="boom")
    with patch("ollama_client.requests.post", return_value=fake_response):
        with pytest.raises(RuntimeError):
            chat([{"role": "user", "content": "Salut"}], stream=False)
