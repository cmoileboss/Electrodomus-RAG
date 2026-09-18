"""Tests unitaires de ChunkService, avec un repository mocké."""

from unittest.mock import MagicMock

from api.services.chunk_service import ChunkService


def _make_service():
    """Crée un ChunkService avec une session et un repository mockés."""
    session = MagicMock()
    service = ChunkService(session)
    service.repository = MagicMock()
    return service, session


def test_get_all_queries_session_directly():
    """get_all() interroge directement la session (pas le repository)."""
    service, session = _make_service()
    session.query.return_value.all.return_value = ["chunk_a", "chunk_b"]

    result = service.get_all()

    assert result == ["chunk_a", "chunk_b"]
    session.query.assert_called_once()


def test_count_delegates_to_repository():
    """count() délègue au repository et retourne son résultat."""
    service, _ = _make_service()
    service.repository.count.return_value = 42

    assert service.count() == 42


def test_get_by_id_delegates_to_repository():
    """get_by_id() délègue au repository et retourne son résultat."""
    service, _ = _make_service()
    fake_chunk = MagicMock()
    service.repository.get_by_id.return_value = fake_chunk

    assert service.get_by_id(1) is fake_chunk
    service.repository.get_by_id.assert_called_once_with(1)


def test_create_delegates_and_returns_chunk():
    """create() transmet les champs au repository et retourne le chunk créé."""
    service, _ = _make_service()
    fake_chunk = MagicMock(id=1)
    service.repository.create.return_value = fake_chunk

    result = service.create(
        document_id=1, chunk_index=0, content="c", embedding_text="e",
        section="Objet", page=1, embedding=[0.1, 0.2],
    )

    assert result is fake_chunk
    service.repository.create.assert_called_once_with(
        document_id=1, chunk_index=0, content="c", embedding_text="e",
        section="Objet", page=1, embedding=[0.1, 0.2],
    )


def test_update_returns_none_when_not_found():
    """update() retourne None et ne commit pas si le chunk est introuvable."""
    service, session = _make_service()
    service.repository.get_by_id.return_value = None

    assert service.update(1, {"content": "x"}) is None
    session.commit.assert_not_called()


def test_update_applies_fields_and_commits():
    """update() applique les champs modifiés et commit la session."""
    service, session = _make_service()
    fake_chunk = MagicMock()
    service.repository.get_by_id.return_value = fake_chunk

    result = service.update(1, {"content": "nouveau contenu"})

    assert fake_chunk.content == "nouveau contenu"
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(fake_chunk)
    assert result is fake_chunk


def test_delete_true_when_repository_confirms():
    """delete() retourne True quand le repository confirme la suppression."""
    service, _ = _make_service()
    service.repository.delete.return_value = True

    assert service.delete(1) is True


def test_delete_false_when_not_found():
    """delete() retourne False quand le chunk est introuvable."""
    service, _ = _make_service()
    service.repository.delete.return_value = False

    assert service.delete(1) is False
