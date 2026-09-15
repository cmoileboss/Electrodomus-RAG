from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k_all_relevant():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0


def test_precision_at_k_partial_match():
    assert precision_at_k([1, 2, 3, 4], {1, 3}, 4) == 0.5


def test_precision_at_k_no_match():
    assert precision_at_k([1, 2, 3], {99}, 3) == 0.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {1, 2}, 5) == 0.0


def test_precision_at_k_uses_only_top_k():
    # Le seul id pertinent (3) est hors du top-2, donc precision@2 = 0
    assert precision_at_k([1, 2, 3], {3}, 2) == 0.0


def test_recall_at_k_all_found():
    assert recall_at_k([1, 2, 3], {1, 2}, 3) == 1.0


def test_recall_at_k_partial_found():
    assert recall_at_k([1, 2], {1, 2, 3}, 2) == 2 / 3


def test_recall_at_k_no_relevant_ids_returns_zero():
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_recall_at_k_respects_k_cutoff():
    assert recall_at_k([1, 2, 3], {3}, 2) == 0.0
    assert recall_at_k([1, 2, 3], {3}, 3) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([1, 2, 3], {3}) == 1 / 3


def test_reciprocal_rank_no_match_returns_zero():
    assert reciprocal_rank([1, 2, 3], {99}) == 0.0
