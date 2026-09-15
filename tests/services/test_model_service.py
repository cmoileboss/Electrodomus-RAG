from unittest.mock import MagicMock

from api.services.model_service import ModelService


def _make_service():
    session = MagicMock()
    service = ModelService(session)
    service.repository = MagicMock()
    return service, session


def test_get_all_delegates_to_repository():
    service, _ = _make_service()
    service.repository.get_all.return_value = ["model_a", "model_b"]

    assert service.get_all() == ["model_a", "model_b"]


def test_get_by_id_delegates_to_repository():
    service, _ = _make_service()
    fake_model = MagicMock()
    service.repository.get_by_id.return_value = fake_model

    assert service.get_by_id(1) is fake_model


def test_create_delegates_and_returns_model():
    service, _ = _make_service()
    fake_model = MagicMock(id=1)
    service.repository.create.return_value = fake_model

    result = service.create(name="FR-600", type="four encastrable")

    assert result is fake_model
    service.repository.create.assert_called_once_with(name="FR-600", type="four encastrable")


def test_update_returns_none_when_not_found():
    service, session = _make_service()
    service.repository.get_by_id.return_value = None

    assert service.update(1, {"name": "x"}) is None
    session.commit.assert_not_called()


def test_update_applies_fields_and_commits():
    service, session = _make_service()
    fake_model = MagicMock()
    service.repository.get_by_id.return_value = fake_model

    result = service.update(1, {"name": "FR-650"})

    assert fake_model.name == "FR-650"
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(fake_model)
    assert result is fake_model


def test_delete_true_when_repository_confirms():
    service, _ = _make_service()
    service.repository.delete.return_value = True

    assert service.delete(1) is True


def test_delete_false_when_not_found():
    service, _ = _make_service()
    service.repository.delete.return_value = False

    assert service.delete(1) is False
