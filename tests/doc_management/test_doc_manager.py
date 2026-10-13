"""Tests de la détection de changement et de la mise à jour des chunks dans DocManager."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from doc_management import doc_manager as dm
from doc_management.doc_manager import DocManager, compute_file_hash


def _model(name, type_="four encastrable"):
    return SimpleNamespace(id=1, name=name, type=type_)


def _manager():
    """DocManager sans chargement des modèles lourds."""
    manager = DocManager.__new__(DocManager)
    manager.converter = MagicMock()
    manager.chunker = MagicMock()
    manager.model = MagicMock()
    return manager


def test_compute_file_hash_depends_on_content(tmp_path):
    """L'empreinte est stable pour un même contenu et change si le contenu change."""
    f = tmp_path / "doc.txt"
    f.write_text("version 1")
    first = compute_file_hash(f)
    assert compute_file_hash(f) == first
    f.write_text("version 2")
    assert compute_file_hash(f) != first


def test_get_or_create_model_returns_existing():
    """Un modèle déjà en base est retourné sans création."""
    repo = MagicMock()
    repo.get_by_name.return_value = _model("FR-600")
    assert DocManager._get_or_create_model(repo, "FR-600").name == "FR-600"
    repo.create.assert_not_called()


def test_get_or_create_model_creates_with_type_of_same_prefix():
    """Un nouveau modèle hérite du type des modèles de même préfixe."""
    repo = MagicMock()
    repo.get_by_name.return_value = None
    repo.get_all.return_value = [_model("FR-600", "four encastrable"), _model("LV-100", "lave-vaisselle")]
    DocManager._get_or_create_model(repo, "FR-900")
    repo.create.assert_called_once_with(name="FR-900", type="four encastrable")


def test_get_or_create_model_unknown_prefix_returns_none():
    """Un préfixe inconnu ne permet pas de déduire le type : aucun modèle n'est créé."""
    repo = MagicMock()
    repo.get_by_name.return_value = None
    repo.get_all.return_value = [_model("FR-600")]
    assert DocManager._get_or_create_model(repo, "ZZ-100") is None
    repo.create.assert_not_called()


def test_register_new_models_creates_only_unknown_known_prefix_models():
    """Seules les références absentes construites sur un préfixe connu sont ajoutées."""
    repo = MagicMock()
    existing = {"FR-600": _model("FR-600")}
    repo.get_all.return_value = list(existing.values())
    repo.get_by_name.side_effect = existing.get
    _manager()._register_new_models(["Le FR-600 et le FR-900, pas le ZZ-123 ni FR-12"], repo)
    repo.create.assert_called_once_with(name="FR-900", type="four encastrable")


def test_process_document_unchanged_is_skipped(tmp_path):
    """Un document dont l'empreinte est inchangée n'est ni converti ni modifié."""
    f = tmp_path / "manuel.txt"
    f.write_text("contenu")
    document = SimpleNamespace(id=1, content_hash=compute_file_hash(f))
    manager = _manager()
    with patch.object(dm, "DocumentRepository") as MockDocRepo:
        MockDocRepo.return_value.get_by_filepath.return_value = document
        assert manager.process_document(str(f), MagicMock()) == "unchanged"
    manager.converter.convert.assert_not_called()


def test_process_document_changed_replaces_chunks_and_date(tmp_path):
    """Un document modifié voit ses chunks remplacés, sa date recherchée de nouveau et son hash mis à jour."""
    f = tmp_path / "manuel.txt"
    f.write_text("Édition juin 2026")
    document = SimpleNamespace(id=7, content_hash="ancien", date="2020", filepath="x")
    session = MagicMock()

    manager = _manager()
    chunk = MagicMock()
    chunk.text = "Édition juin 2026"
    chunk.meta.headings = ["Titre"]
    chunk.meta.doc_items = []
    manager.chunker.chunk.return_value = [chunk]
    manager.chunker.contextualize.return_value = "Édition juin 2026"
    manager.model.encode.return_value = np.zeros((1, 3))

    with patch.object(dm, "DocumentRepository") as MockDocRepo, \
         patch.object(dm, "ChunkRepository"), \
         patch.object(dm, "ErrorRepository") as MockErrRepo, \
         patch.object(dm, "ModelRepository") as MockModelRepo:
        MockDocRepo.return_value.get_by_filepath.return_value = document
        MockErrRepo.return_value.get_all_unique_codes.return_value = []
        MockModelRepo.return_value.get_all.return_value = []

        assert manager.process_document(str(f), session) == "updated"

    session.query.return_value.filter_by.assert_called_with(document_id=7)
    session.query.return_value.filter_by.return_value.delete.assert_called_once()
    assert document.content_hash == compute_file_hash(f)
    assert document.date == "2026-06"


def test_build_referentiel_entries_one_chunk_per_row_with_headers(tmp_path):
    """Chaque ligne de tableau devient un chunk avec ses en-têtes, ses modèles, son code et la date de version."""
    headers = ["Code", "Signification", "Conduite à tenir (client)", "Intervention SAV (technicien)", "Modèles concernés"]
    intro = [["Référentiel des codes erreur — Lave-linge serie WX"] + [None] * 4,
             ["Version 2026-T4 (arrêtée au 15 septembre 2026)"] + [None] * 4,
             [None] * 5]
    wx = pd.DataFrame(intro + [headers,
                               ["E01", "Porte mal verrouillée", "Refermer la porte.", None, "WX-300, WX-350"],
                               ["E10", "Arrivée d'eau", "Ouvrir le robinet.", "Contrôler l'électrovanne.", "WX-300"],
                               [None] * 5])
    lv = pd.DataFrame([["Référentiel LV"] + [None] * 4,
                       ["Version 2025-T4 (arrêtée au 15 décembre 2025)"] + [None] * 4,
                       [None] * 5, headers,
                       ["E01", "Porte mal fermée", "Refermer.", "Contrôler.", "LV-100"]])
    path = tmp_path / "Referentiel_codes_erreur.xlsx"
    with pd.ExcelWriter(path) as writer:
        wx.to_excel(writer, sheet_name="Gamme WX", header=False, index=False)
        lv.to_excel(writer, sheet_name="Gamme LV", header=False, index=False)

    entries, date = _manager()._build_referentiel_entries(str(path))

    assert len(entries) == 3
    first = entries[0]
    assert first.content == (
        "Code : E01\nSignification : Porte mal verrouillée\n"
        "Conduite à tenir (client) : Refermer la porte.\nModèles concernés : WX-300, WX-350"
    )
    assert first.embedding_text.startswith("Référentiel des codes erreur - Lave-linge serie WX\nVersion 2026-T4")
    assert first.embedding_text.endswith(first.content)
    assert first.section == "Gamme WX"
    assert first.models == ["WX-300", "WX-350"]
    assert first.error_codes == ["E01"]
    assert entries[2].section == "Gamme LV"
    assert date == "2026-09-15"
