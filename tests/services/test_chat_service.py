"""Tests unitaires de ChatService : recherche hybride (vecteur + BM25), fusion RRF et génération Ollama."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.services.chat_service import ChatService


class FakeEmbedModel:
    """Modèle d'embedding factice retournant toujours le même vecteur."""

    def encode(self, text, normalize_embeddings=True):
        return np.array([0.1, 0.2, 0.3])


def _make_chunk(chunk_id, section="Objet", page=1, document_id=5, content="contenu"):
    return SimpleNamespace(id=chunk_id, section=section, page=page, document_id=document_id, content=content)


@pytest.mark.asyncio
async def test_embed_calls_encode_with_normalization():
    """_embed() appelle encode() du modèle avec normalize_embeddings=True."""
    service = ChatService(FakeEmbedModel())

    embedding = await service._embed("Ma question")

    assert list(embedding) == [0.1, 0.2, 0.3]


def test_get_nearest_vectorized_chunks_delegates_to_repository():
    """_get_nearest_vectorized_chunks() interroge ChunkRepository.get_nearest via une session."""
    fake_chunk = _make_chunk(1)
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.get_nearest.return_value = [fake_chunk]

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls):
        service = ChatService(FakeEmbedModel())
        result = service._get_nearest_vectorized_chunks(np.array([0.1, 0.2, 0.3]), limit=5)

    assert result == [fake_chunk]
    mock_repo_cls.return_value.get_nearest.assert_called_once_with([0.1, 0.2, 0.3], limit=5)


def test_get_nearest_bm25_chunks_delegates_to_repository():
    """_get_nearest_bm25_chunks() interroge ChunkRepository.search_bm25 via une session."""
    fake_chunk = _make_chunk(2)
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.search_bm25.return_value = [fake_chunk]

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls):
        service = ChatService(FakeEmbedModel())
        result = service._get_nearest_bm25_chunks("Ma question", limit=5)

    assert result == [fake_chunk]
    mock_repo_cls.return_value.search_bm25.assert_called_once_with("Ma question", limit=5)


def test_merge_scores_fuses_by_rrf_and_dedupes():
    """_merge_scores() fusionne vecteur et BM25 par RRF et dédoublonne par id de chunk."""
    c1, c2, c3 = _make_chunk(1), _make_chunk(2), _make_chunk(3)

    merged = ChatService._merge_scores([c1, c2], [c2, c3], limit=3)

    # c2 est en tête (présent dans les deux classements), puis c1 avant c3 (meilleur rang vectoriel).
    assert [c.id for c in merged] == [2, 1, 3]


def test_merge_scores_respects_limit():
    """_merge_scores() tronque le résultat fusionné à la limite demandée."""
    chunks = [_make_chunk(i) for i in range(1, 6)]

    merged = ChatService._merge_scores(chunks, [], limit=2)

    assert len(merged) == 2


def test_build_rag_message_with_chunks():
    """_build_rag_message() inclut le contenu et les métadonnées de source des chunks."""
    chunk = _make_chunk(1, section="Sécurité", page=3, document_id=7, content="Ne pas ouvrir la porte.")

    message = ChatService._build_rag_message("Comment ouvrir ?", [chunk])

    assert "CONTEXTE :" in message
    assert "Sécurité" in message
    assert "page 3" in message
    assert "Ne pas ouvrir la porte." in message
    assert "QUESTION : Comment ouvrir ?" in message


def test_build_rag_message_without_chunks():
    """_build_rag_message() indique l'absence d'extrait pertinent quand la liste est vide."""
    message = ChatService._build_rag_message("Question", [])

    assert "Aucun extrait pertinent trouvé." in message


def test_build_messages_includes_system_history_and_rag_content():
    """_build_messages() assemble le prompt système, l'historique puis le message RAG."""
    history = [SimpleNamespace(role="user", content="Bonjour")]

    messages = ChatService._build_messages("Question", history, "RAG CONTENT")

    assert messages[0]["role"] == "system"
    assert {"role": "user", "content": "Bonjour"} in messages
    assert messages[-1] == {"role": "user", "content": "RAG CONTENT"}


def test_build_sources_extracts_metadata():
    """_build_sources() extrait section, page et document_id de chaque chunk."""
    chunk = _make_chunk(1, section="Objet", page=2, document_id=5)

    sources = ChatService._build_sources([chunk])

    assert sources == [{"section": "Objet", "page": 2, "document_id": 5}]


def test_chat_non_streaming_returns_full_message():
    """_chat() en mode non-streaming retourne directement le contenu du message de la réponse."""
    mock_response = MagicMock(ok=True)
    mock_response.json.return_value = {"message": {"content": "La réponse"}}

    with patch("api.services.chat_service.requests.post", return_value=mock_response) as mock_post:
        result = ChatService._chat([{"role": "user", "content": "Question"}], stream=False)

    assert result == "La réponse"
    assert mock_post.call_args.kwargs["json"]["stream"] is False


def test_chat_streaming_concatenates_tokens():
    """_chat() en mode streaming concatène les tokens jusqu'à l'événement 'done'."""
    lines = [
        b'{"message": {"content": "Bon"}}',
        b'{"message": {"content": "jour"}, "done": true}',
    ]
    mock_response = MagicMock(ok=True)
    mock_response.iter_lines.return_value = lines

    with patch("api.services.chat_service.requests.post", return_value=mock_response):
        result = ChatService._chat([{"role": "user", "content": "Question"}], stream=True)

    assert result == "Bonjour"


def test_chat_raises_on_error_response():
    """_chat() lève une RuntimeError si Ollama répond avec une erreur."""
    mock_response = MagicMock(ok=False, status_code=500, text="erreur serveur")

    with patch("api.services.chat_service.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError):
            ChatService._chat([{"role": "user", "content": "Question"}], stream=False)


@pytest.mark.asyncio
async def test_ask_returns_answer_and_sources():
    """ask() fusionne les résultats vecteur/BM25 et retourne la réponse générée avec ses sources."""
    fake_chunk = _make_chunk(1, section="Objet", page=2, document_id=5)
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.get_nearest.return_value = [fake_chunk]
    mock_repo_cls.return_value.search_bm25.return_value = []

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls), \
         patch.object(ChatService, "_chat", return_value="La réponse") as mock_chat:
        service = ChatService(FakeEmbedModel())
        result = await service.ask("Ma question", [], limit=5)

    assert result["answer"] == "La réponse"
    assert result["sources"] == [{"section": "Objet", "page": 2, "document_id": 5}]
    mock_chat.assert_called_once()
    assert mock_chat.call_args.args[1] is False


@pytest.mark.asyncio
async def test_ask_includes_history_in_messages_sent_to_chat():
    """ask() inclut l'historique de conversation dans les messages envoyés à _chat()."""
    history = [SimpleNamespace(role="user", content="Bonjour")]
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.get_nearest.return_value = []
    mock_repo_cls.return_value.search_bm25.return_value = []

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls), \
         patch.object(ChatService, "_chat", return_value="ok") as mock_chat:
        service = ChatService(FakeEmbedModel())
        await service.ask("Question", history, limit=3)

    sent_messages = mock_chat.call_args.args[0]
    assert {"role": "user", "content": "Bonjour"} in sent_messages
    assert sent_messages[-1]["role"] == "user"
    assert "QUESTION : Question" in sent_messages[-1]["content"]
