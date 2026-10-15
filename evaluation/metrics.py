"""Métriques de retrieval : Precision@K, Recall@K, Reciprocal Rank."""


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Calcule la précision parmi les résultats effectivement retournés dans le top-k.

    Args:
        retrieved_ids: Identifiants récupérés, classés du plus pertinent au moins pertinent.
        relevant_ids: Ensemble des identifiants considérés comme pertinents.
        k: Nombre maximal de résultats à évaluer.

    Returns:
        float: Fraction des résultats du top-k qui sont pertinents; ``0.0`` si le
            top-k ne contient aucun résultat.
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Calcule la proportion des éléments pertinents retrouvés dans le top-k.

    Args:
        retrieved_ids: Identifiants récupérés, classés par pertinence décroissante.
        relevant_ids: Ensemble des identifiants pertinents attendus.
        k: Nombre maximal de résultats à évaluer.

    Returns:
        float: Fraction des identifiants pertinents retrouvés; ``0.0`` si aucun
            identifiant pertinent n'est attendu.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Calcule l'inverse du rang du premier résultat pertinent.

    Args:
        retrieved_ids: Identifiants récupérés, classés par pertinence décroissante.
        relevant_ids: Ensemble des identifiants considérés comme pertinents.

    Returns:
        float: Inverse du rang (à partir de 1) du premier résultat pertinent, ou
            ``0.0`` si aucun résultat pertinent n'est récupéré.
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0
