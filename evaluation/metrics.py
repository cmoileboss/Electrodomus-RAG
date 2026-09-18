"""Métriques de retrieval : Precision@K, Recall@K, Reciprocal Rank."""


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Proportion des k premiers résultats récupérés qui sont pertinents."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Proportion des chunks pertinents retrouvés parmi les k premiers résultats."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Inverse du rang du premier résultat pertinent (0 si aucun)."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0
