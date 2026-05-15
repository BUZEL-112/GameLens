"""
Evaluation Entry Point

This module provides a comprehensive suite to validate the quality and performance of the trained Two-Tower model.
It ensures that the model provides personalized lift over baseline strategies and maintains high semantic relevance.

Evaluation Phases:
1. Artifact Restoration: Loads the saved Keras towers, FAISS indices, and precomputed similarity tables.
2. Test Stratification: Isolates a held-out fraction of users (default 10%) who meet minimum interaction thresholds.
3. Embedding Projection: Generates 128-dim latent vectors for test users using the finalized User Tower.
4. Accuracy Assessment: Computes Recall@K (retrieval coverage) and NDCG@K (ranking quality) for multiple K values.
5. Popularity Benchmarking: Compares model results against a 'Most Popular' baseline to calculate personalized lift.
6. Embedding Sanity: Validates vector normalization, checks for 'dead' dimensions, and tests genre-based clustering.
7. Semantic Consistency: Uses Tag Jaccard similarity and historical Co-play rates to verify recommendation logic.

The final summary provides a clear PASS/FAIL gate based on whether the model outperforms the popularity baseline.
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf

from training.utils.utils import load_config
from training.utils.logger import logger
from training.pipeline import stage1_split_sides, stage2_process_items, stage3_process_users


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_recall_at_k(
    all_scores: np.ndarray,
    test_user_ids: list[str],
    test_user_positives: dict[str, set],
    idx_to_name: dict[int, str],
    K: int = 20,
) -> float:
    """
    Measures the fraction of relevant items that appear in the top K recommendations.
    Recall = (Relevant Items Recommended) / (Total Relevant Items)
    """
    recalls = []
    for i, uid in enumerate(test_user_ids):
        positives = test_user_positives.get(uid, set())
        if not positives:
            continue
        # Use argpartition for efficient top-K retrieval without full sorting
        top_k_names = {
            idx_to_name[idx] for idx in np.argpartition(all_scores[i], -K)[-K:]
        }
        recall = len(positives & top_k_names) / len(positives)
        recalls.append(recall)
    
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    logger.info(f"  Recall@{K:<3}: {mean_recall:.4f}  ({len(recalls)} users)")
    return mean_recall


def popularity_baseline(
    interactions_df: pd.DataFrame,
    test_user_positives: dict[str, set],
    K: int = 20,
) -> float:
    """
    Calculates Recall@K for a non-personalized 'Most Popular' strategy.
    This serves as the 'floor' that the personalized model MUST beat.
    """
    play_totals = (
        interactions_df.groupby("item_name")["interaction"]
        .sum()
        .sort_values(ascending=False)
    )
    # Recommendations are simply the top K most played games globally
    top_k_names = set(play_totals.head(K).index.tolist())
    
    recalls = []
    for uid, positives in test_user_positives.items():
        if not positives:
            continue
        recall = len(positives & top_k_names) / len(positives)
        recalls.append(recall)
    
    baseline = float(np.mean(recalls)) if recalls else 0.0
    logger.info(f"  Popularity@{K:<3}: {baseline:.4f}")
    return baseline


def ndcg_at_k(
    user_scores: np.ndarray,
    positives_set: set,
    idx_to_name: dict[int, str],
    K: int = 20,
) -> float:
    """
    Normalized Discounted Cumulative Gain. 
    Unlike Recall, NDCG rewards relevant items appearing higher in the recommendation list.
    """
    # Sort top K by actual score
    top_k_idx = np.argpartition(user_scores, -K)[-K:]
    top_k_idx = top_k_idx[np.argsort(user_scores[top_k_idx])[::-1]]

    # DCG = sum of relevance / log(rank)
    dcg = sum(
        (1 / np.log2(rank + 2))
        for rank, idx in enumerate(top_k_idx)
        if idx_to_name.get(idx) in positives_set
    )
    
    # IDCG is the score of a perfect ranking
    n_ideal = min(len(positives_set), K)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_tag_jaccard(
    similarity_table: dict,
    items_df: pd.DataFrame,
    n_sample: int = 500,
) -> tuple[float, list[float]]:
    """
    Evaluates item-to-item quality by checking how many tags overlap between neighbors.
    A high Jaccard score (>0.3) indicates the model has learned semantic similarities.
    """
    tag_lookup = {row["item_name"]: set(row["tags"]) for _, row in items_df.iterrows()}
    jaccards = []
    sample_items = list(similarity_table.keys())[:n_sample]

    for item_name in sample_items:
        query_tags = tag_lookup.get(item_name, set())
        if not query_tags:
            continue
        # Compare tags with each nearest neighbor retrieved from the embedding space
        for neighbor in similarity_table[item_name]:
            nb_tags = tag_lookup.get(neighbor["item_name"], set())
            if not nb_tags:
                continue
            union = query_tags | nb_tags
            if union:
                jaccards.append(len(query_tags & nb_tags) / len(union))

    mean_j = float(np.mean(jaccards)) if jaccards else 0.0
    logger.info(f"  Tag Jaccard (n={n_sample}): {mean_j:.3f}  "
                f"({'PASS' if mean_j > 0.3 else 'FAIL'})")
    return mean_j, jaccards


def evaluate_coplay_consistency(
    similarity_table: dict,
    interactions_df: pd.DataFrame,
    n_sample: int = 200,
) -> tuple[float, list[float]]:
    """
    Checks if recommended items are frequently played together in real-world history.
    Consistency = (Recommended Items in Co-play History) / (Total Recommendations)
    """
    logger.info("  Building co-play sets...")
    # Map user -> set of games they've played
    user_histories = (
        interactions_df[interactions_df["interaction"] > 0.01]
        .groupby("user_id")["item_name"]
        .apply(set)
        .to_dict()
    )

    sample_items = list(similarity_table.keys())[:n_sample]
    coplay_sets = {item: set() for item in sample_items}
    sample_set = set(sample_items)

    # Invert the mapping: Item -> set of all games played by anyone who played this item
    for history in user_histories.values():
        for item in history & sample_set:
            coplay_sets[item].update(history - {item})

    rates = []
    for item_name in sample_items:
        coplay = coplay_sets.get(item_name, set())
        neighbors = similarity_table.get(item_name, [])
        neighbor_names = {n["item_name"] for n in neighbors}
        if coplay and neighbor_names:
            # How many of the embedding-based neighbors actually appear in history?
            rate = len(neighbor_names & coplay) / len(neighbor_names)
            rates.append(rate)

    mean_rate = float(np.mean(rates)) if rates else 0.0
    logger.info(f"  Co-play consistency (n={n_sample}): {mean_rate:.3f}  "
                f"({'PASS' if mean_rate > 0.4 else 'FAIL'})")
    return mean_rate, rates


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    save_dir = config.get("serving", {}).get("artifacts_path", "model_artifacts")

    # ── Phase 1: Load saved artifacts ─────────────────────────────────────────
    logger.info("Loading artifacts...")
    with open(f"{save_dir}/artifacts.pkl", "rb") as f:
        art = pickle.load(f)

    item_vocab = art["item_vocab"]
    idx_to_name = art["idx_to_name"]
    all_user_vectors = art["all_user_vectors"]
    num_items = art["num_items"]

    # Restore Keras models and served indices
    u_tower = tf.keras.models.load_model(f"{save_dir}/u_tower.keras")
    item_embeddings = np.load(f"{save_dir}/item_embeddings.npy")
    index = faiss.read_index(f"{save_dir}/item_index.faiss")

    with open(f"{save_dir}/similarity_table.pkl", "rb") as f:
        similarity_table = pickle.load(f)

    logger.info(f"  {num_items:,} items, {len(all_user_vectors):,} users loaded")

    # ── Phase 2: Load source data for evaluation context ───────────────────────
    feat_cfg = config.get("feature_engineering", {})
    transformed_path = feat_cfg.get("transformed_data_path", "data/processed/transformed_df.csv")
    df = pd.read_csv(transformed_path)
    items_df, interactions_df = stage1_split_sides(df)
    item_data = stage2_process_items(items_df)

    # ── Phase 3: Create held-out test split ─────────────────────────────────────
    test_fraction = eval_cfg.get("test_fraction", 0.10)
    min_pos = eval_cfg.get("min_pos_per_user", 5)
    eval_threshold = eval_cfg.get("eval_threshold", 0.5)

    all_uids = np.array(list(all_user_vectors.keys()))
    np.random.seed(42)
    np.random.shuffle(all_uids)

    n_test = int(len(all_uids) * test_fraction)
    test_uid_set = set(all_uids[:n_test].tolist())

    test_user_positives = {}
    test_user_vectors = {}

    for uid, group in interactions_df.groupby("user_id", sort=False):
        if uid not in test_uid_set:
            continue
        # Only evaluate on high-confidence 'liked' games
        pos_names = set(
            group.loc[group["interaction"] > eval_threshold, "item_name"].tolist()
        )
        pos_names &= set(item_vocab.keys())
        # Ensure test users have enough history to make evaluation meaningful
        if len(pos_names) >= min_pos:
            test_user_positives[uid] = pos_names
            test_user_vectors[uid] = all_user_vectors[uid]

    logger.info(
        f"Test users: {len(test_user_positives):,}  "
        f"(min {min_pos} positives each)"
    )

    # ── Phase 4: Project test users into embedding space ────────────────────────
    EMB_BATCH = 1024
    test_user_ids = list(test_user_positives.keys())
    user_emb_chunks = []
    for start in range(0, len(test_user_ids), EMB_BATCH):
        end = min(start + EMB_BATCH, len(test_user_ids))
        batch = np.stack([test_user_vectors[uid] for uid in test_user_ids[start:end]])
        chunk = u_tower(batch, training=False).numpy()
        user_emb_chunks.append(chunk)
    user_embeddings = np.vstack(user_emb_chunks)

    # Generate similarity scores for every (test_user, all_items) pair
    all_scores = user_embeddings @ item_embeddings.T
    logger.info(f"Score matrix: {all_scores.shape}")

    # ── Phase 5: Recall@K Benchmarking ──────────────────────────────────────────
    logger.info("\n== Recall@K ==")
    recall_scores = {}
    for K in eval_cfg.get("recall_k_values", [10, 20, 50, 100]):
        recall_scores[K] = evaluate_recall_at_k(
            all_scores, test_user_ids, test_user_positives, idx_to_name, K=K
        )

    # ── Phase 6: Popularity Lift Analysis ────────────────────────────────────────
    logger.info("\n== Popularity Baseline ==")
    baseline_recalls = {}
    for K in eval_cfg.get("recall_k_values", [10, 20, 50, 100]):
        baseline_recalls[K] = popularity_baseline(
            interactions_df, test_user_positives, K=K
        )

    logger.info("\n== Lift over Baseline ==")
    passed = True
    for K in recall_scores:
        model = recall_scores[K]
        base = baseline_recalls[K]
        lift_pct = (model - base) / max(base, 1e-9) * 100
        status = "PASS" if lift_pct > 0 else "FAIL"
        logger.info(
            f"  K={K:<3}  model={model:.4f}  baseline={base:.4f}  "
            f"lift={lift_pct:+.1f}%  [{status}]"
        )
        if lift_pct <= 0:
            passed = False

    logger.info(f"\nGate: {'PASSED' if passed else 'FAILED'}")

    # ── Phase 7: Ranking Quality (NDCG@K) ───────────────────────────────────────
    logger.info("\n== NDCG@K ==")
    ndcg_scores = {}
    for K in eval_cfg.get("ndcg_k_values", [10, 20, 50]):
        scores_k = []
        for i, uid in enumerate(test_user_ids):
            positives = test_user_positives.get(uid, set())
            if not positives:
                continue
            scores_k.append(ndcg_at_k(all_scores[i], positives, idx_to_name, K=K))
        ndcg_scores[K] = float(np.mean(scores_k)) if scores_k else 0.0
        logger.info(f"  NDCG@{K:<3}: {ndcg_scores[K]:.4f}")

    # ── Phase 8: Embedding Sanity Checks ────────────────────────────────────────
    logger.info("\n== Embedding Sanity Checks ==")

    # Clustering Check: Are FPS games close to other FPS games?
    fps_games = ["Counter-Strike", "Team Fortress Classic", "Day of Defeat",
                 "Half-Life", "Quake Live"]
    fps_indices = [item_vocab[g] for g in fps_games if g in item_vocab]
    if fps_indices:
        fps_embs = item_embeddings[fps_indices]
        fps_sim = fps_embs @ fps_embs.T
        np.fill_diagonal(fps_sim, 0)
        avg_fps_sim = fps_sim.sum() / max(len(fps_indices) * (len(fps_indices) - 1), 1)
        logger.info(f"  Avg FPS-to-FPS similarity: {avg_fps_sim:.3f}  (want > 0.3)")

    # Variance Check: Are we utilizing all dimensions or have we collapsed into a few?
    per_dim_var = item_embeddings.var(axis=0)
    dead_dims = int((per_dim_var < 1e-4).sum())
    logger.info(f"  Dead embedding dimensions: {dead_dims}/128  (want < 10)")

    # Norm Check: Ensure the UnitNormalization layer is correctly enforcing L2 distance
    norms = np.linalg.norm(item_embeddings, axis=1)
    norm_ok = bool(np.allclose(norms, 1.0, atol=1e-5))
    logger.info(f"  All item norms == 1.0: {norm_ok}")

    # ── Phase 9: Item-to-Item Semantic Quality ──────────────────────────────────
    logger.info("\n== Item-to-Item Quality ==")
    mean_jaccard, _ = evaluate_tag_jaccard(
        similarity_table, item_data["items_df"], n_sample=500
    )
    mean_coplay, _ = evaluate_coplay_consistency(
        similarity_table, interactions_df, n_sample=200
    )

    # ── Final Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Evaluation Summary")
    logger.info("=" * 55)
    for K in recall_scores:
        logger.info(f"  Recall@{K}: {recall_scores[K]:.4f}")
    for K in ndcg_scores:
        logger.info(f"  NDCG@{K}:   {ndcg_scores[K]:.4f}")
    logger.info(f"  Tag Jaccard:       {mean_jaccard:.3f}")
    logger.info(f"  Co-play rate:      {mean_coplay:.3f}")
    logger.info(f"  Baseline gate:     {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 55)

    # ── Write Metrics Manifest ────────────────────────────────────────────────
    manifest = {
        "timestamp": time.time(),
        "baseline_gate_passed": passed,
        "recall_scores": recall_scores,
        "ndcg_scores": ndcg_scores,
        "tag_jaccard": mean_jaccard,
        "coplay_rate": mean_coplay
    }
    
    manifest_dir = config.get("orchestration", {}).get("manifest_dir", "model_artifacts/manifests")
    import os
    import json
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "metrics.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Two-Tower recommender")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()
    main(config_path=args.config)
