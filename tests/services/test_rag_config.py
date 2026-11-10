"""Tests des réglages RAG modifiables par requête : validation, endpoint /config et application dans ChatService."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.routers import config as config_router
from api.services.chat_service import ChatService
from api.services.rag_config import ALLOWED_MODELS, MAX_CHUNK_LIMIT, RagConfig


def test_allowed_models_include_both_gpt_oss():
    """Les deux modèles gpt-oss sont proposés."""
    assert {"openai/gpt-oss-20b", "openai/gpt-oss-120b"} <= set(ALLOWED_MODELS)


def test_rag_config_rejects_unknown_model():
    """Un modèle hors liste autorisée est refusé."""
    with pytest.raises(ValidationError):
        RagConfig(model="openai/autre-modele")


@pytest.mark.parametrize("field", ["vector_limit", "bm25_limit", "rrf_limit", "rerank_top_k"])
@pytest.mark.parametrize("value", [0, -1, MAX_CHUNK_LIMIT + 1])
def test_rag_config_rejects_out_of_range_limits(field, value):
    """Les nombres de chunks doivent être compris entre 1 et MAX_CHUNK_LIMIT."""
    with pytest.raises(ValidationError):
        RagConfig(**{field: value})


def test_config_endpoint_returns_defaults_models_and_max_limit():
    """GET /config expose les valeurs par défaut, les modèles et la borne maximale."""
    app = FastAPI()
    app.include_router(config_router.router)
    body = TestClient(app).get("/config/").json()

    assert body["defaults"] == RagConfig().model_dump()
    assert body["models"] == ALLOWED_MODELS
    assert body["max_limit"] == MAX_CHUNK_LIMIT


@pytest.mark.asyncio
async def test_ask_applies_config_to_each_step():
    """ask() applique les limites et le modèle de la config à chaque étape du pipeline."""
    chunks = [MagicMock(id=i, section="s", page=1, document_id=1, content="c") for i in range(1, 6)]
    mock_repo_cls = MagicMock()
    mock_repo_cls.return_value.get_nearest.return_value = chunks
    mock_repo_cls.return_value.search_bm25.return_value = chunks
    reranker = MagicMock()
    reranker.rerank.return_value = chunks[:2]
    embed_model = MagicMock()
    embed_model.encode.return_value = np.array([0.1, 0.2])
    config = RagConfig(model="openai/gpt-oss-120b", vector_limit=7, bm25_limit=8, rrf_limit=4, rerank_top_k=2)

    with patch("api.services.chat_service.get_session", return_value=MagicMock()), \
         patch("api.services.chat_service.ChunkRepository", mock_repo_cls), \
         patch("api.services.chat_service.DocumentRepository"), \
         patch("api.services.chat_service._write_prompt_to_file"), \
         patch("api.services.chat_service._write_response_to_file"), \
         patch.object(ChatService, "_chat", return_value="ok") as mock_chat:
        await ChatService(embed_model, reranker).ask("Question", [], config=config)

    assert mock_repo_cls.return_value.get_nearest.call_args.kwargs["limit"] == 7
    assert mock_repo_cls.return_value.search_bm25.call_args.kwargs["limit"] == 8
    assert len(reranker.rerank.call_args.args[1]) == 4
    assert reranker.rerank.call_args.kwargs["top_k"] == 2
    assert mock_chat.call_args.args[2] == "openai/gpt-oss-120b"
