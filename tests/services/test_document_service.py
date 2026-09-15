from unittest.mock import MagicMock

from api.services.document_service import DocumentService


def _make_service():
    session = MagicMock()
    service = DocumentService(session)
    service.repository = MagicMock()
    return service, session


def test_get_all_delegates_to_repository():
    service, _ = _make_service()
    service.repository.get_all.return_value = ["doc_a", "doc_b"]

    assert service.get_all() == ["doc_a", "doc_b"]


def test_get_by_id_delegates_to_repository():
    service, _ = _make_service()
    fake_doc = MagicMock()
    service.repository.get_by_id.return_value = fake_doc

    assert service.get_by_id(1) is fake_doc


def test_create_delegates_and_returns_document():
    service, _ = _make_service()
    fake_doc = MagicMock(id=1)
    service.repository.create.return_value = fake_doc

    result = service.create(title="Titre", filepath="/tmp/doc.pdf")

    assert result is fake_doc
    service.repository.create.assert_called_once_with(
        title="Titre", filepath="/tmp/doc.pdf", date=None
    )


def test_update_returns_none_when_not_found():
    service, session = _make_service()
    service.repository.get_by_id.return_value = None

    assert service.update(1, {"title": "x"}) is None
    session.commit.assert_not_called()


def test_update_applies_fields_and_commits():
    service, session = _make_service()
    fake_doc = MagicMock()
    service.repository.get_by_id.return_value = fake_doc

    result = service.update(1, {"title": "Nouveau titre"})

    assert fake_doc.title == "Nouveau titre"
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(fake_doc)
    assert result is fake_doc


def test_delete_true_when_repository_confirms():
    service, _ = _make_service()
    service.repository.delete.return_value = True

    assert service.delete(1) is True


def test_delete_false_when_not_found():
    service, _ = _make_service()
    service.repository.delete.return_value = False

    assert service.delete(1) is False
