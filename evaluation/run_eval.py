"""Évalue le retrieval et la génération du RAG (Precision@K, Recall@K, MRR) sur un jeu de questions.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --k 1 3 5 10
    python -m evaluation.run_eval --output evaluation/results/run.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from database.chunk_repository import ChunkRepository
from database.database import get_session
from evaluation.dataset import EVAL_QUESTIONS
from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from logger import get_logger
from api.services.chat_service import SYSTEM_PROMPT, ChatService

load_dotenv()
logger = get_logger(__name__)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN")
RESULTS_DIR = Path(__file__).parent / "results"


def run_evaluation(k_values: list[int], output_path: Path) -> None:
    """Exécute l'évaluation sur EVAL_QUESTIONS et écrit le rapport en console et en JSON."""
    max_k = max(k_values)
    embed_model = SentenceTransformer(EMBED_MODEL_ID, use_auth_token=HF_TOKEN)
    generated_at = datetime.now().isoformat(timespec="seconds")

    per_question_results = []
    with get_session() as session:
        repo = ChunkRepository(session)
        for item in EVAL_QUESTIONS:
            question = item["question"]
            relevant_ids = set(item["relevant_chunk_ids"])
            if not relevant_ids:
                logger.warning("Question sans chunks pertinents renseignés, ignorée : %s", question)
                continue

            embedding = embed_model.encode(question, normalize_embeddings=True).tolist()
            retrieved = repo.get_nearest(embedding, limit=max_k)
            retrieved_ids = [c.id for c in retrieved]

            rag_content = build_rag_message(question, retrieved)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rag_content},
            ]
            answer = chat(messages, stream=False)

            per_question_results.append({
                "question": question,
                "expected_chunk_ids": sorted(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": {
                    "precision": {str(k): precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
                    "recall": {str(k): recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
                    "rr": reciprocal_rank(retrieved_ids, relevant_ids),
                },
                "answer": answer,
                "generated_at": generated_at,
            })

    if not per_question_results:
        logger.warning("Aucune question évaluable : renseignez 'relevant_chunk_ids' dans evaluation/dataset.py")
        return

    _print_report(per_question_results, k_values)
    _write_json(per_question_results, k_values, generated_at, output_path)


def _print_report(results: list[dict], k_values: list[int]) -> None:
    """Affiche dans la console les métriques par question et leurs moyennes."""
    n = len(results)
    for r in results:
        print(f"\nQ: {r['question']}")
        for k in k_values:
            print(f"  Precision@{k}: {r['metrics']['precision'][str(k)]:.2f}  Recall@{k}: {r['metrics']['recall'][str(k)]:.2f}")
        print(f"  RR: {r['metrics']['rr']:.2f}")

    print("\n=== Moyennes ===")
    for k in k_values:
        mean_precision = sum(r["metrics"]["precision"][str(k)] for r in results) / n
        mean_recall = sum(r["metrics"]["recall"][str(k)] for r in results) / n
        print(f"Precision@{k}: {mean_precision:.3f}  Recall@{k}: {mean_recall:.3f}")
    mrr = sum(r["metrics"]["rr"] for r in results) / n
    print(f"MRR: {mrr:.3f}  (sur {n} question(s) évaluée(s))")


def _write_json(results: list[dict], k_values: list[int], generated_at: str, output_path: Path) -> None:
    """Écrit les résultats détaillés et le résumé des métriques dans un fichier JSON."""
    n = len(results)
    summary = {
        "precision": {str(k): sum(r["metrics"]["precision"][str(k)] for r in results) / n for k in k_values},
        "recall": {str(k): sum(r["metrics"]["recall"][str(k)] for r in results) / n for k in k_values},
        "mrr": sum(r["metrics"]["rr"] for r in results) / n,
    }
    payload = {
        "generated_at": generated_at,
        "k_values": k_values,
        "summary": summary,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Résultats écrits dans %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du retrieval RAG.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="Valeurs de K à évaluer")
    parser.add_argument("--output", type=str, default=None, help="Chemin du fichier JSON de résultats")
    args = parser.parse_args()

    default_output = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_evaluation(sorted(set(args.k)), Path(args.output) if args.output else default_output)
