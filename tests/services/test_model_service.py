"""Tests unitaires de ModelService, avec un repository mocké."""

from unittest.mock import MagicMock

from api.services.model_service import ModelService


def _make_service():
    """Crée un ModelService avec une session et un repository mockés."""
    session = MagicMock()
    service = ModelService(session)
    service.repository = MagicMock()
    return service, session


def test_get_all_delegates_to_repository():
    """get_all() délègue au repository et retourne son résultat."""
    service, _ = _make_service()
    service.repository.get_all.return_value = ["model_a", "model_b"]

    assert service.get_all() == ["model_a", "model_b"]


def test_get_by_id_delegates_to_repository():
    """get_by_id() délègue au repository et retourne son résultat."""
    service, _ = _make_service()
    fake_model = MagicMock()
    service.repository.get_by_id.return_value = fake_model

    assert service.get_by_id(1) is fake_model


def test_create_delegates_and_returns_model():
    """create() transmet les champs au repository et retourne le modèle créé."""
    service, _ = _make_service()
    fake_model = MagicMock(id=1)
    service.repository.create.return_value = fake_model

    result = service.create(name="FR-600", type="four encastrable")

    assert result is fake_model
    service.repository.create.assert_called_once_with(name="FR-600", type="four encastrable")


def test_update_returns_none_when_not_found():
    """update() retourne None et ne commit pas si le modèle est introuvable."""
    service, session = _make_service()
    service.repository.get_by_id.return_value = None

    assert service.update(1, {"name": "x"}) is None
    session.commit.assert_not_called()


def test_update_applies_fields_and_commits():
    """update() applique les champs modifiés et commit la session."""
    service, session = _make_service()
    fake_model = MagicMock()
    service.repository.get_by_id.return_value = fake_model

    result = service.update(1, {"name": "FR-650"})

    assert fake_model.name == "FR-650"
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(fake_model)
    assert result is fake_model


def test_delete_true_when_repository_confirms():
    """delete() retourne True quand le repository confirme la suppression."""
    service, _ = _make_service()
    service.repository.delete.return_value = True

    assert service.delete(1) is True


def test_delete_false_when_not_found():
    """delete() retourne False quand le modèle est introuvable."""
    service, _ = _make_service()
    service.repository.delete.return_value = False

    assert service.delete(1) is False
