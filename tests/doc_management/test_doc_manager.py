"""Tests de la détection de changement et de la mise à jour des chunks dans DocManager."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from database.models import Chunk, Model
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


@pytest.mark.parametrize("filename, expected", [
    ("Gamme_FR_manuel.pdf", ["four encastrable"]),
    ("Manuel_Gamme_LV.pdf", ["lave-vaisselle"]),
    ("Gamme_WX.PDF", ["lave-linge"]),
    ("gamme_wx_notice.pdf", ["lave-linge"]),
    ("Gamme_FR.docx", []),
    ("Gamme_FR.txt", []),
    ("Manuel_FR-600.pdf", []),
    ("Guide des fours encastrables.docx", ["four encastrable"]),
    ("Entretien_four.txt", ["four encastrable"]),
    ("FAQ_Lave-linge.html", ["lave-linge"]),
    ("Notice_lave_vaisselle_2026.pdf", ["lave-vaisselle"]),
    ("Guide lave linge et lave-vaisselle.pdf", ["lave-linge", "lave-vaisselle"]),
    ("Gamme_FR_four.pdf", ["four encastrable"]),
    ("Contrat_fournisseur.pdf", []),
    ("PROC-FR-001_Nettoyage.pdf", ["four encastrable"]),
    ("proc-lv_vidange.docx", ["lave-vaisselle"]),
    ("PROC-WX_Entretien.txt", ["lave-linge"]),
    ("PROC-WX-12.html", ["lave-linge"]),
    ("Annexe_PROC-FR.pdf", []),
    ("PROC-XX_divers.pdf", []),
])
def test_document_model_types_from_name(filename, expected):
    """Les types d'appareils sont déduits d'un mot du titre (four, lave-linge, lave-vaisselle) ou d'un PDF Gamme_*."""
    assert DocManager._document_model_types(Path(filename)) == expected


def _ingest_new_document(tmp_path, filename):
    """Ingère un nouveau document de deux chunks et retourne les noms de modèles liés à chaque chunk."""
    f = tmp_path / filename
    f.write_bytes(b"contenu")
    manager = _manager()
    chunks = []
    for text in ["Premier chunk", "Second chunk mentionnant le LV-100"]:
        chunk = MagicMock()
        chunk.text = text
        chunk.meta.headings = ["Titre"]
        chunk.meta.doc_items = []
        chunks.append(chunk)
    manager.chunker.chunk.return_value = chunks
    manager.chunker.contextualize.side_effect = lambda chunk: chunk.text
    manager.model.encode.return_value = np.zeros((2, 3))

    models = {
        m.name: m for m in [
            Model(id=1, name="FR-600", type="four encastrable"), Model(id=2, name="FR-650", type="four encastrable"),
            Model(id=3, name="LV-100", type="lave-vaisselle"), Model(id=4, name="WX-300", type="lave-linge"),
        ]
    }
    session = MagicMock()
    with patch.object(dm, "DocumentRepository") as MockDocRepo, \
         patch.object(dm, "ChunkRepository"), \
         patch.object(dm, "ErrorRepository") as MockErrRepo, \
         patch.object(dm, "ModelRepository") as MockModelRepo:
        MockDocRepo.return_value.get_by_filepath.return_value = None
        MockDocRepo.return_value.get_by_title.return_value = None
        MockErrRepo.return_value.get_all_unique_codes.return_value = []
        MockModelRepo.return_value.get_all.return_value = list(models.values())
        MockModelRepo.return_value.get_by_name.side_effect = models.get
        assert manager.process_document(str(f), session) == "created"

    added = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], Chunk)]
    return [sorted(m.name for m in chunk.models) for chunk in added]


def test_gamme_pdf_links_all_models_of_the_type_to_every_chunk(tmp_path):
    """Un PDF Gamme_FR relie tous les fours à chaque chunk, en plus des modèles détectés dans le texte."""
    linked = _ingest_new_document(tmp_path, "Gamme_FR_manuel.pdf")
    assert linked == [["FR-600", "FR-650"], ["FR-600", "FR-650", "LV-100"]]


def test_non_gamme_document_only_links_detected_models(tmp_path):
    """Sans nom de gamme, seuls les modèles cités dans le texte sont reliés."""
    linked = _ingest_new_document(tmp_path, "manuel.pdf")
    assert linked == [[], ["LV-100"]]


def test_title_with_lave_linge_links_all_washing_machines_to_every_chunk(tmp_path):
    """Un titre contenant « lave-linge » relie tous les lave-linge à chaque chunk, quelle que soit l'extension."""
    linked = _ingest_new_document(tmp_path, "FAQ_lave-linge.txt")
    assert linked == [["WX-300"], ["LV-100", "WX-300"]]


@pytest.mark.parametrize("title, model, expected", [
    ("Manuel_FR-600", "FR-600", True),
    ("Manuel_FR600_v2", "FR-600", True),
    ("Notice fr 600", "FR-600", True),
    ("Manuel_FR_600", "FR-600", True),
    ("Manuel_FR-6000", "FR-600", False),
    ("Manuel_FR-60", "FR-600", False),
    ("Manuel_FR-650", "FR-600", False),
    ("Manuel_AFR-600", "FR-600", False),
    ("Gamme_FR", "FR-600", False),
])
def test_title_mentions_model(title, model, expected):
    """Le titre doit contenir la référence complète du modèle, pas un préfixe ni un autre numéro."""
    assert DocManager._title_mentions_model(title, model) is expected


def test_model_in_file_name_is_linked_to_every_chunk(tmp_path):
    """Un fichier dont le nom contient FR-600 relie uniquement ce modèle à chaque chunk."""
    linked = _ingest_new_document(tmp_path, "Manuel_FR-600.pdf")
    assert linked == [["FR-600"], ["FR-600", "LV-100"]]


def test_proc_prefix_links_all_models_of_the_type_to_every_chunk(tmp_path):
    """Un fichier PROC-LV... relie tous les lave-vaisselle à chaque chunk."""
    linked = _ingest_new_document(tmp_path, "PROC-LV-004_Vidange.pdf")
    assert linked == [["LV-100"], ["LV-100"]]
