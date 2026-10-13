"""Tests de process_all : sélection des fichiers nouveaux ou modifiés, référentiel traité en premier."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from doc_management import chunks_storing as cs
from doc_management.doc_manager import compute_file_hash


@contextmanager
def _fake_session():
    yield MagicMock()


def _run(tmp_path, documents):
    processed = []
    with patch.object(cs, "DOC_FOLDER", tmp_path), \
         patch.object(cs, "get_session", _fake_session), \
         patch.object(cs, "DocumentRepository") as MockRepo, \
         patch.object(cs, "build_doc_manager"), \
         patch.object(cs, "_process_file", side_effect=lambda f, m: processed.append(f)):
        MockRepo.return_value.get_all.return_value = documents
        cs.process_all()
    return [Path(f).name for f in processed]


def test_process_all_skips_unchanged_and_processes_new_and_changed(tmp_path):
    """Seuls les fichiers nouveaux, modifiés ou sans hash enregistré sont ingérés."""
    unchanged = tmp_path / "a.txt"
    unchanged.write_text("a")
    changed = tmp_path / "b.txt"
    changed.write_text("b nouveau")
    legacy = tmp_path / "c.txt"
    legacy.write_text("c")
    new = tmp_path / "d.txt"
    new.write_text("d")
    documents = [
        SimpleNamespace(filepath=str(unchanged), content_hash=compute_file_hash(unchanged)),
        SimpleNamespace(filepath=str(changed), content_hash="ancien"),
        SimpleNamespace(filepath=str(legacy), content_hash=None),
    ]
    assert sorted(_run(tmp_path, documents)) == ["b.txt", "c.txt", "d.txt"]


def test_process_all_processes_referentiel_first(tmp_path):
    """Le référentiel des codes d'erreur est traité avant les autres documents."""
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "Referentiel_codes_erreur.txt").write_text("r")
    (tmp_path / "z.txt").write_text("z")
    assert _run(tmp_path, [])[0] == "Referentiel_codes_erreur.txt"
