"""
Training Pipeline — Stages 1 through 6

This module provides the core functional blocks for the Two-Tower model training pipeline.
It transforms a consolidated interaction dataset into high-dimensional latent embeddings.

The pipeline execution logic is broken into six distinct stages:
1. Side Splitting: Separates the 'item' features (metadata) from 'user' interaction records.
2. Item Processing: Builds vocabularies and a multi-hot content matrix for genres and tags.
3. User Processing: Synthesizes 322-dim user vectors representing weighted historical interests.
4. Pair Construction: Samples positive user-item pairs and generates synthetic hard and random negatives.
5. Tensor Assembly: Packages data into high-performance NumPy arrays ready for the TensorFlow engine.
6. Training Loop: Executes deep learning with InfoNCE loss, in-batch negatives, and cosine LR decay.

Each stage is designed as a standalone function for improved testability and modularity.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from training.utils.logger import logger
from core_ml.features import safe_parse, build_user_vector, normalize_user_matrix


# safe_parse is imported from core_ml.features — it is the canonical implementation
# shared with the serving API. Do not redefine it here.


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Separating the Two Sides
# ═══════════════════════════════════════════════════════════════════════════

def stage1_split_sides(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decouples the unified dataset into a catalog of unique items and a table of user interactions.

    Step 1: Normalize engagement scores (interaction) based on log-playtime.
    Step 2: Extract unique item metadata (genres, tags).
    Step 3: Clean and isolate user interaction records (user_id, item_name, interaction).
    """
    # Ensure interaction scores are calculated and normalized
    if "interaction" not in df.columns:
        logger.info("Creating 'interaction' column from playtime...")
        df = df.copy()
        df["playtime"] = pd.to_numeric(df["playtime"], errors="coerce")
        df = df.dropna(subset=["playtime"])
        df = df.drop_duplicates(subset=["user_id", "item_id"])
        
        # log1p transformation dampens the effect of extreme playtime outliers
        df["log_playtime"] = np.log1p(df["playtime"])
        
        # Normalize interactions per-user so active/inactive users are on a comparable scale (0-5)
        df["max_log_playtime"] = df.groupby("user_id")["log_playtime"].transform("max")
        df["interaction"] = (df["log_playtime"] / df["max_log_playtime"]) * 5
        df = df.drop(columns=["log_playtime", "max_log_playtime"])

    # --- LEFT side: Unique Item Features ---
    items_df = (
        df[["item_id", "item_name", "genres", "tags"]]
        .drop_duplicates(subset=["item_id"])
        .reset_index(drop=True)
    )
    items_df["genres"] = items_df["genres"].apply(safe_parse)
    items_df["tags"] = items_df["tags"].apply(safe_parse)

    # --- RIGHT side: User-Item Interaction Map ---
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
    Builds indices and sparse matrices representing the game catalog.

    Step 1: Create a unified Item Vocabulary based on item names.
    Step 2: Generate frequency-based vocabularies for Genres and Tags.
    Step 3: Construct a multi-hot content matrix for all games.
    """
    # Standardize on item_name as the primary join key
    items_df = items_df.drop_duplicates(subset=["item_name"]).reset_index(drop=True)

    item_vocab = {name: idx for idx, name in enumerate(items_df["item_name"])}
    idx_to_name = {v: k for k, v in item_vocab.items()}
    num_items = len(item_vocab)

    # Restrict Genre and Tag features to top-N most frequent to reduce dimensionality
    all_genres = [g for genres in items_df["genres"] for g in genres]
    genre_vocab = {
        g: i for i, (g, _) in enumerate(Counter(all_genres).most_common(num_genres))
    }

    all_tags = [t for tags in items_df["tags"] for t in tags]
    tag_vocab = {
        t: i for i, (t, _) in enumerate(Counter(all_tags).most_common(num_tags))
    }

    # Initialize matrices for multi-hot encoding
    genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
    tag_matrix = np.zeros((num_items, num_tags), dtype=np.float32)

    # Populate matrices based on game metadata
    for _, row in items_df.iterrows():
        idx = item_vocab[row["item_name"]]
        for g in row["genres"]:
            if g in genre_vocab:
                genre_matrix[idx, genre_vocab[g]] = 1.0
        for t in row["tags"]:
            if t in tag_vocab:
                tag_matrix[idx, tag_vocab[t]] = 1.0

    # Concatenate genre and tag flags into a single content vector per item
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
    Summarizes user behavior into a fixed-length 322-dimensional feature vector.

    Layout:
    - [0-319]: Weighted average of item content vectors from the user's history.
    - [320]:   Normalized total interaction score (volume of play).
    - [321]:   Normalized total number of games played (diversity of play).

    Delegates to core_ml.features.build_user_vector() for the per-user
    computation, ensuring mathematical parity with the nearline updater.
    """
    logger.info("Precomputing user vectors...")
    raw_vectors: dict[str, np.ndarray] = {}
    for uid, group in interactions_df.groupby("user_id", sort=False):
        raw_vectors[uid] = build_user_vector(group, item_content_matrix, item_vocab)
    logger.info(f"  {len(raw_vectors):,} users cached.")

    # Apply global max normalization across all users (corpus-level, training-time only)
    all_user_vectors = normalize_user_matrix(raw_vectors)

    user_feat_dim = next(iter(all_user_vectors.values())).shape[0]
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
    """
    Finds 'Hard Negatives'—items that look like what the user plays but haven't been played.
    These are critical for teaching the model to distinguish between general interest and specific choices.
    """
    if not played_indices:
        return np.random.randint(0, num_items, size=n)

    # Compute the user's preference centroid based on played content
    user_centroid = item_content_matrix[played_indices].mean(axis=0)
    
    # Score all items by cosine similarity to the centroid
    all_scores = item_content_matrix @ user_centroid
    # Mask out items already played
    all_scores[played_indices] = -np.inf

    # Pick top scoring non-played items
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
    Generates the final training dataset using a triplet-like sampling strategy.

    For each positive interaction, we sample:
    1. Hard Negatives: Games with similar genres/tags that were NOT played.
    2. Random Negatives: Arbitrary games from the catalog to provide baseline contrast.
    """
    all_item_arr = np.array(list(item_vocab.keys()))

    pos_rows = []
    neg_rows = []

    for uid, group in interactions_df.groupby("user_id", sort=False):
        # Identify confirmed likes
        pos = group[group["interaction"] > min_interaction]
        if len(pos) == 0:
            continue

        played_names = set(group["item_name"].tolist())
        played_idx = [item_vocab[n] for n in played_names if n in item_vocab]

        # Record Positives (Label 1)
        pos_rows.extend(
            (uid, name, 1, float(w))
            for name, w in zip(pos["item_name"], pos["interaction"])
            if name in item_vocab
        )

        # Record Hard Negatives (Label 0)
        hard_idx = _sample_hard_negatives(
            played_idx, item_content_matrix, num_items,
            n=len(pos) * neg_per_pos_hard,
        )
        neg_rows.extend(
            (uid, idx_to_name[i], 0, 1.0)
            for i in hard_idx
            if i in idx_to_name and idx_to_name[i] not in played_names
        )

        # Record Random 'Easy' Negatives (Label 0)
        oversample = min(
            len(pos) * neg_per_pos_random + len(played_names) + 100, num_items
        )
        sampled = np.random.choice(all_item_arr, size=oversample, replace=False)
        rand_neg = sampled[~np.isin(sampled, list(played_names))][
            : len(pos) * neg_per_pos_random
        ]
        neg_rows.extend((uid, name, 0, 1.0) for name in rand_neg)

    # Combine and shuffle for stochastic gradient descent
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
    Formats sampled pairs into structured NumPy arrays for the TensorFlow training loop.
    This stage avoids expensive lookups during training by pre-assembling all necessary vectors.
    """
    ts = train_samples.iloc[:n_train] if n_train else train_samples

    y = ts["label"].to_numpy(dtype=np.float32)
    weights = ts["weight"].to_numpy(dtype=np.float32)

    # Retrieve precomputed user vectors
    _zero_user = np.zeros(user_feat_dim, dtype=np.float32)
    user_feat = np.stack(
        [all_user_vectors.get(uid, _zero_user) for uid in ts["user_id"]],
    ).astype(np.float32)

    # Retrieve item IDs and their corresponding content features
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
    Optimizes tower weights using in-batch contrastive learning (InfoNCE).

    Logic:
    - Every positive User-Item pair in a batch is treated as the 'target'.
    - Every OTHER item in that same batch acts as a negative sample for that user.
    - InfoNCE maximizes the similarity of the true pair while minimizing similarity to batch-negatives.
    """
    y = tensors["y"]
    w = tensors["weights"]
    uf = tensors["user_feat"]
    ii = tensors["item_ids"]
    if_ = tensors["item_feat"]

    # Filter for positive interactions only (standard approach for InfoNCE)
    pos_mask = y == 1
    uf_pos = uf[pos_mask]
    ii_pos = ii[pos_mask]
    if_pos = if_[pos_mask]
    w_pos = w[pos_mask]

    n_pos = len(uf_pos)
    num_batches = n_pos // batch_size

    # Cosine learning rate decay for stable convergence
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
        # Shuffle positives each epoch
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
                # Forward pass through both towers
                u_emb = u_tower(u_b, training=True)
                i_emb = i_tower([i_id_b, i_b], training=True)

                # Compute Similarity Matrix (Batch x Batch)
                # Diagonal = Positive pairs; Off-diagonal = Negative pairs
                sim_matrix = tf.matmul(u_emb, i_emb, transpose_b=True) / temperature
                labels_matrix = tf.eye(B)
                
                # Cross-entropy objective for multi-class classification (target = diagonal)
                row_loss = tf.keras.losses.categorical_crossentropy(
                    labels_matrix, sim_matrix, from_logits=True
                )
                loss = tf.reduce_mean(row_loss * w_b)

            # Backpropagation
            grads = tape.gradient(loss, all_vars)
            optimizer.apply_gradients(zip(grads, all_vars))
            epoch_loss += float(loss)

        avg_loss = epoch_loss / max(num_batches, 1)

        # Performance Monitoring: Score a fixed probe batch to track embedding quality
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
