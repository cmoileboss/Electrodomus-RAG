"""Tests des métriques de retrieval (precision@k, recall@k, reciprocal rank)."""

from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k_all_relevant():
    """Precision@k vaut 1 quand tous les résultats retournés sont pertinents."""
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0


def test_precision_at_k_partial_match():
    """Precision@k reflète la proportion de résultats pertinents."""
    assert precision_at_k([1, 2, 3, 4], {1, 3}, 4) == 0.5


def test_precision_at_k_no_match():
    """Precision@k vaut 0 quand aucun résultat n'est pertinent."""
    assert precision_at_k([1, 2, 3], {99}, 3) == 0.0


def test_precision_at_k_empty_retrieved():
    """Precision@k vaut 0 quand aucun résultat n'est retourné."""
    assert precision_at_k([], {1, 2}, 5) == 0.0


def test_precision_at_k_uses_only_top_k():
    """Precision@k ne considère que les k premiers résultats."""
    # Le seul id pertinent (3) est hors du top-2, donc precision@2 = 0
    assert precision_at_k([1, 2, 3], {3}, 2) == 0.0


def test_recall_at_k_all_found():
    """Recall@k vaut 1 quand tous les ids pertinents sont retrouvés."""
    assert recall_at_k([1, 2, 3], {1, 2}, 3) == 1.0


def test_recall_at_k_partial_found():
    """Recall@k reflète la proportion d'ids pertinents retrouvés."""
    assert recall_at_k([1, 2], {1, 2, 3}, 2) == 2 / 3


def test_recall_at_k_no_relevant_ids_returns_zero():
    """Recall@k vaut 0 quand il n'y a aucun id pertinent attendu."""
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_recall_at_k_respects_k_cutoff():
    """Recall@k ne considère que les k premiers résultats."""
    assert recall_at_k([1, 2, 3], {3}, 2) == 0.0
    assert recall_at_k([1, 2, 3], {3}, 3) == 1.0


def test_reciprocal_rank_first_position():
    """Le reciprocal rank vaut 1 quand le premier résultat est pertinent."""
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0


def test_reciprocal_rank_third_position():
    """Le reciprocal rank vaut 1/rang pour le premier résultat pertinent trouvé."""
    assert reciprocal_rank([1, 2, 3], {3}) == 1 / 3


def test_reciprocal_rank_no_match_returns_zero():
    """Le reciprocal rank vaut 0 quand aucun résultat n'est pertinent."""
    assert reciprocal_rank([1, 2, 3], {99}) == 0.0
