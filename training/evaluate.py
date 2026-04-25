"""
Evaluation Entry Point

Loads saved model artifacts and runs the full evaluation suite:
    - Recall@K at multiple K values
    - Popularity baseline comparison (pass/fail gate)
    - NDCG@K (ranking quality)
    - Embedding sanity checks (norms, dead dims, genre clustering)
    - Item-to-item quality: tag Jaccard, co-play consistency

Usage:
    cd game_recommender
    python -m training.evaluate
    python -m training.evaluate --config configs/config.yaml
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
    recalls = []
    for i, uid in enumerate(test_user_ids):
        positives = test_user_positives.get(uid, set())
        if not positives:
            continue
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
    play_totals = (
        interactions_df.groupby("item_name")["interaction"]
        .sum()
        .sort_values(ascending=False)
    )
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
    top_k_idx = np.argpartition(user_scores, -K)[-K:]
    top_k_idx = top_k_idx[np.argsort(user_scores[top_k_idx])[::-1]]

    dcg = sum(
        (1 / np.log2(rank + 2))
        for rank, idx in enumerate(top_k_idx)
        if idx_to_name.get(idx) in positives_set
    )
    n_ideal = min(len(positives_set), K)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_tag_jaccard(
    similarity_table: dict,
    items_df: pd.DataFrame,
    n_sample: int = 500,
) -> tuple[float, list[float]]:
    tag_lookup = {row["item_name"]: set(row["tags"]) for _, row in items_df.iterrows()}
    jaccards = []
    sample_items = list(similarity_table.keys())[:n_sample]

    for item_name in sample_items:
        query_tags = tag_lookup.get(item_name, set())
        if not query_tags:
            continue
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
    logger.info("  Building co-play sets...")
    user_histories = (
        interactions_df[interactions_df["interaction"] > 0.01]
        .groupby("user_id")["item_name"]
        .apply(set)
        .to_dict()
    )

    sample_items = list(similarity_table.keys())[:n_sample]
    coplay_sets = {item: set() for item in sample_items}
    sample_set = set(sample_items)

    for history in user_histories.values():
        for item in history & sample_set:
            coplay_sets[item].update(history - {item})

    rates = []
    for item_name in sample_items:
        coplay = coplay_sets.get(item_name, set())
        neighbors = similarity_table.get(item_name, [])
        neighbor_names = {n["item_name"] for n in neighbors}
        if coplay and neighbor_names:
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

    # ── Load artifacts ────────────────────────────────────────────────────────
    logger.info("Loading artifacts...")
    with open(f"{save_dir}/artifacts.pkl", "rb") as f:
        art = pickle.load(f)

    item_vocab = art["item_vocab"]
    idx_to_name = art["idx_to_name"]
    all_user_vectors = art["all_user_vectors"]
    num_items = art["num_items"]
    user_feat_dim = art["USER_FEAT_DIM"]
    embedding_dim = art["EMBEDDING_DIM"]

    u_tower = tf.keras.models.load_model(f"{save_dir}/u_tower.keras")
    i_tower = tf.keras.models.load_model(f"{save_dir}/i_tower.keras")
    item_embeddings = np.load(f"{save_dir}/item_embeddings.npy")
    index = faiss.read_index(f"{save_dir}/item_index.faiss")

    with open(f"{save_dir}/similarity_table.pkl", "rb") as f:
        similarity_table = pickle.load(f)

    logger.info(f"  {num_items:,} items, {len(all_user_vectors):,} users loaded")

    # ── Load source data for evaluation context ──────────────────────────────
    feat_cfg = config.get("feature_engineering", {})
    transformed_path = feat_cfg.get("transformed_data_path", "data/processed/transformed_df.csv")
    df = pd.read_csv(transformed_path)
    items_df, interactions_df = stage1_split_sides(df)
    item_data = stage2_process_items(items_df)

    # ── Create held-out test split ────────────────────────────────────────────
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
        pos_names = set(
            group.loc[group["interaction"] > eval_threshold, "item_name"].tolist()
        )
        pos_names &= set(item_vocab.keys())
        if len(pos_names) >= min_pos:
            test_user_positives[uid] = pos_names
            test_user_vectors[uid] = all_user_vectors[uid]

    logger.info(
        f"Test users: {len(test_user_positives):,}  "
        f"(min {min_pos} positives each)"
    )

    # ── Extract embeddings ────────────────────────────────────────────────────
    EMB_BATCH = 1024
    test_user_ids = list(test_user_positives.keys())
    user_emb_chunks = []
    for start in range(0, len(test_user_ids), EMB_BATCH):
        end = min(start + EMB_BATCH, len(test_user_ids))
        batch = np.stack([test_user_vectors[uid] for uid in test_user_ids[start:end]])
        chunk = u_tower(batch, training=False).numpy()
        user_emb_chunks.append(chunk)
    user_embeddings = np.vstack(user_emb_chunks)

    all_scores = user_embeddings @ item_embeddings.T
    logger.info(f"Score matrix: {all_scores.shape}")

    # ── Recall@K ──────────────────────────────────────────────────────────────
    logger.info("\n== Recall@K ==")
    recall_scores = {}
    for K in eval_cfg.get("recall_k_values", [10, 20, 50, 100]):
        recall_scores[K] = evaluate_recall_at_k(
            all_scores, test_user_ids, test_user_positives, idx_to_name, K=K
        )

    # ── Popularity Baseline ───────────────────────────────────────────────────
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

    logger.info(
        f"\nGate: {'PASSED' if passed else 'FAILED'}"
    )

    # ── NDCG@K ────────────────────────────────────────────────────────────────
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

    # ── Embedding Sanity Checks ───────────────────────────────────────────────
    logger.info("\n== Embedding Sanity Checks ==")

    fps_games = ["Counter-Strike", "Team Fortress Classic", "Day of Defeat",
                 "Half-Life", "Quake Live"]
    fps_indices = [item_vocab[g] for g in fps_games if g in item_vocab]
    if fps_indices:
        fps_embs = item_embeddings[fps_indices]
        fps_sim = fps_embs @ fps_embs.T
        np.fill_diagonal(fps_sim, 0)
        avg_fps_sim = fps_sim.sum() / max(len(fps_indices) * (len(fps_indices) - 1), 1)
        logger.info(f"  Avg FPS-to-FPS similarity: {avg_fps_sim:.3f}  (want > 0.3)")

    per_dim_var = item_embeddings.var(axis=0)
    dead_dims = int((per_dim_var < 1e-4).sum())
    logger.info(f"  Dead embedding dimensions: {dead_dims}/128  (want < 10)")

    norms = np.linalg.norm(item_embeddings, axis=1)
    norm_ok = bool(np.allclose(norms, 1.0, atol=1e-5))
    logger.info(f"  All item norms == 1.0: {norm_ok}")

    # ── Item-to-Item Quality ──────────────────────────────────────────────────
    logger.info("\n== Item-to-Item Quality ==")
    mean_jaccard, _ = evaluate_tag_jaccard(
        similarity_table, item_data["items_df"], n_sample=500
    )
    mean_coplay, _ = evaluate_coplay_consistency(
        similarity_table, interactions_df, n_sample=200
    )

    # ── Summary ───────────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Two-Tower recommender")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()
    main(config_path=args.config)
