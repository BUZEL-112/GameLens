"""
Training Pipeline — Stages 1 through 6

Converts a merged CSV of user-item interactions into trained Two-Tower
embeddings.  Each stage is a standalone function that takes explicit inputs
and returns explicit outputs so they can be composed, tested, or re-run
individually.

Stages:
    1. split_sides        — Separate items_df and interactions_df
    2. process_items      — Build item vocab, genre/tag vocab, content matrix
    3. process_users      — Aggregate user feature vectors (322-dim)
    4. build_training_pairs — Positives + hard/random negatives
    5. assemble_tensors   — Numpy arrays ready for training
    6. train_loop         — InfoNCE with cosine LR decay
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from training.utils.logger import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_parse(val) -> list:
    """Safely parse list-like columns that may be stored as strings."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Separating the Two Sides
# ═══════════════════════════════════════════════════════════════════════════

def stage1_split_sides(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the merged dataframe into:
      - items_df:        unique items with genres/tags  (LEFT side — what a game IS)
      - interactions_df: user-item interaction records  (RIGHT side — who a user IS)

    The incoming ``df`` must already have an ``interaction`` column
    (normalised log-playtime).  If it only has ``playtime``, this function
    creates the interaction column.
    """
    # Create interaction column if missing
    if "interaction" not in df.columns:
        logger.info("Creating 'interaction' column from playtime...")
        df = df.copy()
        df["playtime"] = pd.to_numeric(df["playtime"], errors="coerce")
        df = df.dropna(subset=["playtime"])
        df = df.drop_duplicates(subset=["user_id", "item_id"])
        df["log_playtime"] = np.log1p(df["playtime"])
        df["max_log_playtime"] = df.groupby("user_id")["log_playtime"].transform("max")
        df["interaction"] = (df["log_playtime"] / df["max_log_playtime"]) * 5
        df = df.drop(columns=["log_playtime", "max_log_playtime"])

    # --- LEFT: Item features ---
    items_df = (
        df[["item_id", "item_name", "genres", "tags"]]
        .drop_duplicates(subset=["item_id"])
        .reset_index(drop=True)
    )
    items_df["genres"] = items_df["genres"].apply(safe_parse)
    items_df["tags"] = items_df["tags"].apply(safe_parse)

    # --- RIGHT: User interaction records ---
    interactions_df = (
        df[["user_id", "item_name", "interaction"]]
        .dropna(subset=["user_id", "item_name", "interaction"])
        .reset_index(drop=True)
    )
    interactions_df["user_id"] = interactions_df["user_id"].astype(str)
    interactions_df["interaction"] = interactions_df["interaction"].astype(float)

    logger.info(
        f"Stage 1 done — {len(items_df):,} unique items, "
        f"{len(interactions_df):,} interaction records, "
        f"{interactions_df['user_id'].nunique():,} unique users"
    )
    return items_df, interactions_df


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Processing the Item Side
# ═══════════════════════════════════════════════════════════════════════════

def stage2_process_items(
    items_df: pd.DataFrame,
    num_genres: int = 20,
    num_tags: int = 300,
) -> dict[str, Any]:
    """
    Build item vocabulary, genre/tag vocabularies, and multi-hot content matrix.

    Returns a dict with keys:
        item_vocab, genre_vocab, tag_vocab, idx_to_name,
        item_content_matrix, num_items, items_df (de-duped on item_name)
    """
    # De-duplicate on item_name (the join key used everywhere)
    items_df = items_df.drop_duplicates(subset=["item_name"]).reset_index(drop=True)

    item_vocab = {name: idx for idx, name in enumerate(items_df["item_name"])}
    idx_to_name = {v: k for k, v in item_vocab.items()}
    num_items = len(item_vocab)

    # Genre vocabulary (top N by frequency)
    all_genres = [g for genres in items_df["genres"] for g in genres]
    genre_vocab = {
        g: i for i, (g, _) in enumerate(Counter(all_genres).most_common(num_genres))
    }

    # Tag vocabulary (top N by frequency)
    all_tags = [t for tags in items_df["tags"] for t in tags]
    tag_vocab = {
        t: i for i, (t, _) in enumerate(Counter(all_tags).most_common(num_tags))
    }

    # Build multi-hot content matrix
    genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
    tag_matrix = np.zeros((num_items, num_tags), dtype=np.float32)

    for _, row in items_df.iterrows():
        idx = item_vocab[row["item_name"]]
        for g in row["genres"]:
            if g in genre_vocab:
                genre_matrix[idx, genre_vocab[g]] = 1.0
        for t in row["tags"]:
            if t in tag_vocab:
                tag_matrix[idx, tag_vocab[t]] = 1.0

    item_content_matrix = np.hstack([genre_matrix, tag_matrix])

    content_dim = num_genres + num_tags
    logger.info(
        f"Stage 2 done — {num_items:,} items, "
        f"{content_dim}-dim content matrix "
        f"({num_genres} genres + {num_tags} tags)"
    )

    return {
        "item_vocab": item_vocab,
        "genre_vocab": genre_vocab,
        "tag_vocab": tag_vocab,
        "idx_to_name": idx_to_name,
        "item_content_matrix": item_content_matrix,
        "num_items": num_items,
        "items_df": items_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — Processing the User Side
# ═══════════════════════════════════════════════════════════════════════════

def stage3_process_users(
    interactions_df: pd.DataFrame,
    item_content_matrix: np.ndarray,
    item_vocab: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Aggregate each user's interaction history into a single 322-dim vector.

    Vector layout: [weighted_avg_content(320) | total_interaction_norm(1) | num_games_norm(1)]

    Returns dict mapping user_id -> ndarray of shape (322,).
    """
    feat_dim = item_content_matrix.shape[1]

    def _get_user_features(user_history: pd.DataFrame) -> np.ndarray:
        valid = [
            (item_vocab[name], weight)
            for name, weight in zip(user_history["item_name"], user_history["interaction"])
            if name in item_vocab and weight > 0
        ]
        if not valid:
            return np.zeros(feat_dim + 2, dtype=np.float32)

        indices, weights = zip(*valid)
        weights = np.array(weights, dtype=np.float32)
        vecs = item_content_matrix[list(indices)]

        history_vec = (vecs * weights[:, None]).sum(axis=0) / (weights.sum() + 1e-9)
        total_interaction = float(weights.sum())
        num_games = float(len(valid))

        return np.concatenate([history_vec, [total_interaction, num_games]]).astype(
            np.float32
        )

    logger.info("Precomputing user vectors...")
    all_user_vectors: dict[str, np.ndarray] = {}
    for uid, group in interactions_df.groupby("user_id", sort=False):
        all_user_vectors[uid] = _get_user_features(group)
    logger.info(f"  {len(all_user_vectors):,} users cached.")

    # Globally normalise the 2 stat dimensions
    user_ids_list = list(all_user_vectors.keys())
    user_matrix = np.stack([all_user_vectors[uid] for uid in user_ids_list])

    max_total = user_matrix[:, -2].max() + 1e-9
    max_ngames = user_matrix[:, -1].max() + 1e-9
    user_matrix[:, -2] /= max_total
    user_matrix[:, -1] /= max_ngames

    for i, uid in enumerate(user_ids_list):
        all_user_vectors[uid] = user_matrix[i]

    user_feat_dim = user_matrix.shape[1]
    logger.info(f"Stage 3 done — USER_FEAT_DIM = {user_feat_dim}")

    return all_user_vectors


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — Constructing Training Pairs
# ═══════════════════════════════════════════════════════════════════════════

def _sample_hard_negatives(
    played_indices: list[int],
    item_content_matrix: np.ndarray,
    num_items: int,
    n: int = 4,
) -> np.ndarray:
    """Find items with similar content to what the user plays but hasn't played."""
    if not played_indices:
        return np.random.randint(0, num_items, size=n)

    user_centroid = item_content_matrix[played_indices].mean(axis=0)
    all_scores = item_content_matrix @ user_centroid
    all_scores[played_indices] = -np.inf

    top_k = min(50, num_items - len(played_indices))
    top_k = max(top_k, n)
    top_idx = np.argsort(all_scores)[::-1][:top_k]
    return np.random.choice(top_idx, size=min(n, len(top_idx)), replace=False)


def stage4_build_training_pairs(
    interactions_df: pd.DataFrame,
    item_vocab: dict[str, int],
    idx_to_name: dict[int, str],
    item_content_matrix: np.ndarray,
    num_items: int,
    neg_per_pos_hard: int = 2,
    neg_per_pos_random: int = 2,
    min_interaction: float = 0.01,
) -> pd.DataFrame:
    """
    Build (user, item, label, weight) training samples with hard + random negatives.
    """
    all_item_arr = np.array(list(item_vocab.keys()))

    pos_rows = []
    neg_rows = []

    for uid, group in interactions_df.groupby("user_id", sort=False):
        pos = group[group["interaction"] > min_interaction]
        if len(pos) == 0:
            continue

        played_names = set(group["item_name"].tolist())
        played_idx = [item_vocab[n] for n in played_names if n in item_vocab]

        # Positives
        pos_rows.extend(
            (uid, name, 1, float(w))
            for name, w in zip(pos["item_name"], pos["interaction"])
            if name in item_vocab
        )

        # Hard negatives
        hard_idx = _sample_hard_negatives(
            played_idx, item_content_matrix, num_items,
            n=len(pos) * neg_per_pos_hard,
        )
        neg_rows.extend(
            (uid, idx_to_name[i], 0, 1.0)
            for i in hard_idx
            if i in idx_to_name and idx_to_name[i] not in played_names
        )

        # Random negatives
        oversample = min(
            len(pos) * neg_per_pos_random + len(played_names) + 100, num_items
        )
        sampled = np.random.choice(all_item_arr, size=oversample, replace=False)
        rand_neg = sampled[~np.isin(sampled, list(played_names))][
            : len(pos) * neg_per_pos_random
        ]
        neg_rows.extend((uid, name, 0, 1.0) for name in rand_neg)

    cols = ["user_id", "item_name", "label", "weight"]
    train_samples = pd.DataFrame(pos_rows + neg_rows, columns=cols)
    train_samples = train_samples.sample(frac=1, random_state=42).reset_index(drop=True)

    n_pos = (train_samples["label"] == 1).sum()
    n_neg = (train_samples["label"] == 0).sum()
    logger.info(
        f"Stage 4 done — {n_pos:,} positives, {n_neg:,} negatives, "
        f"{len(train_samples):,} total"
    )
    return train_samples


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 — Assemble Training Tensors
# ═══════════════════════════════════════════════════════════════════════════

def stage5_assemble_tensors(
    train_samples: pd.DataFrame,
    all_user_vectors: dict[str, np.ndarray],
    item_vocab: dict[str, int],
    item_content_matrix: np.ndarray,
    user_feat_dim: int = 322,
    n_train: int | None = 500_000,
) -> dict[str, np.ndarray]:
    """
    Convert training samples into numpy arrays suitable for the training loop.

    Returns dict with keys:
        y, weights, user_feat, item_ids, item_feat
    """
    ts = train_samples.iloc[:n_train] if n_train else train_samples

    y = ts["label"].to_numpy(dtype=np.float32)
    weights = ts["weight"].to_numpy(dtype=np.float32)

    _zero_user = np.zeros(user_feat_dim, dtype=np.float32)
    user_feat = np.stack(
        [all_user_vectors.get(uid, _zero_user) for uid in ts["user_id"]],
    ).astype(np.float32)

    item_ids = np.array(
        [item_vocab.get(name, 0) for name in ts["item_name"]], dtype=np.int32
    )
    item_feat = item_content_matrix[item_ids]

    logger.info(
        f"Stage 5 done — {len(y):,} samples assembled "
        f"(user_feat={user_feat.shape}, item_feat={item_feat.shape})"
    )

    return {
        "y": y,
        "weights": weights,
        "user_feat": user_feat,
        "item_ids": item_ids,
        "item_feat": item_feat,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6 — Training Loop (InfoNCE with Cosine LR)
# ═══════════════════════════════════════════════════════════════════════════

def stage6_train_loop(
    u_tower: tf.keras.Model,
    i_tower: tf.keras.Model,
    tensors: dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 512,
    temperature: float = 0.1,
    lr_initial: float = 1e-3,
    lr_floor: float = 1e-5,
) -> dict[str, list[float]]:
    """
    Train with in-batch negatives (InfoNCE loss).

    Uses only positive pairs -- every other item in the batch serves as
    a negative for each user (B-1 free negatives per sample).

    Returns training history dict with keys: loss, pos_score, neg_score.
    """
    y = tensors["y"]
    w = tensors["weights"]
    uf = tensors["user_feat"]
    ii = tensors["item_ids"]
    if_ = tensors["item_feat"]

    # Extract positive pairs only
    pos_mask = y == 1
    uf_pos = uf[pos_mask]
    ii_pos = ii[pos_mask]
    if_pos = if_[pos_mask]
    w_pos = w[pos_mask]

    n_pos = len(uf_pos)
    num_batches = n_pos // batch_size

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr_initial,
        decay_steps=num_batches * epochs,
        alpha=lr_floor,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    logger.info(
        f"Stage 6 — Training: {n_pos:,} positives, "
        f"batch={batch_size}, epochs={epochs}, T={temperature}"
    )

    history: dict[str, list[float]] = {"loss": [], "pos_score": [], "neg_score": []}

    for epoch in range(epochs):
        idx = np.random.permutation(n_pos)
        uf_s = uf_pos[idx]
        ii_s = ii_pos[idx]
        if_s = if_pos[idx]
        w_s = w_pos[idx]

        epoch_loss = 0.0
        all_vars = u_tower.trainable_variables + i_tower.trainable_variables

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            B = end - start

            u_b = uf_s[start:end]
            i_id_b = ii_s[start:end].reshape(-1, 1)
            i_b = if_s[start:end]
            w_b = tf.cast(w_s[start:end], tf.float32)

            with tf.GradientTape() as tape:
                u_emb = u_tower(u_b, training=True)
                i_emb = i_tower([i_id_b, i_b], training=True)

                sim_matrix = tf.matmul(u_emb, i_emb, transpose_b=True) / temperature
                labels_matrix = tf.eye(B)
                row_loss = tf.keras.losses.categorical_crossentropy(
                    labels_matrix, sim_matrix, from_logits=True
                )
                loss = tf.reduce_mean(row_loss * w_b)

            grads = tape.gradient(loss, all_vars)
            optimizer.apply_gradients(zip(grads, all_vars))
            epoch_loss += float(loss)

        avg_loss = epoch_loss / max(num_batches, 1)

        # Probe scores on a mixed pos+neg sample
        probe = min(2_000, len(y))
        u_p = u_tower(uf[:probe], training=False)
        i_p = i_tower([ii[:probe].reshape(-1, 1), if_[:probe]], training=False)
        s_p = tf.reduce_sum(u_p * i_p, axis=1).numpy()
        pm = y[:probe].astype(bool)
        avg_pos = float(s_p[pm].mean()) if pm.any() else float("nan")
        avg_neg = float(s_p[~pm].mean()) if (~pm).any() else float("nan")
        cur_lr = float(lr_schedule(optimizer.iterations))

        history["loss"].append(avg_loss)
        history["pos_score"].append(avg_pos)
        history["neg_score"].append(avg_neg)

        logger.info(
            f"  Epoch {epoch + 1:>3}/{epochs}  "
            f"loss={avg_loss:.4f}  pos={avg_pos:.3f}  neg={avg_neg:.3f}  "
            f"lr={cur_lr:.2e}"
        )

    logger.info("Training complete.")
    return history
