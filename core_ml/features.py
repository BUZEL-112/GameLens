"""
core_ml.features — Online-safe feature transformation functions.

This module contains ONLY logic that is safe to call at inference time.
There is no I/O, no train/test splitting, no sampling, and no TensorFlow
dependency. Every function here has an identical mathematical contract
whether called from training/pipeline.py or recommendation_api/services/nearline.py.

Functions:
  safe_parse(val)
      Parse a column value that may be a list object, a stringified list, or garbage.
      Used on 'genres' and 'tags' columns from both raw GZIP data and Redis-stored metadata.

  compute_log_playtime_interaction(playtime_series, user_id_series)
      Convert raw playtime values into per-user normalized interaction scores (0-5 scale).
      Mirrors the normalization in stage1_split_sides.

  build_user_vector(user_history, item_content_matrix, item_vocab)
      Construct a single user's 322-dim feature vector from their play history.
      This is the critical shared function — the EXACT same computation runs
      in training (Stage 3) and in nearline updates after new events arrive.

  normalize_user_matrix(user_vectors_dict)
      Apply global max normalization to the last two statistical dimensions
      across a corpus of users. Must be called on ALL users together to ensure
      the normalization constants are consistent with training.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd


# ── parse helpers ─────────────────────────────────────────────────────────────

def safe_parse(val) -> list:
    """
    Safely parse a value that may be a list, a stringified Python list, or garbage.

    Handles:
    - Actual list objects → returned as-is
    - Strings like "['Action', 'RPG']" → evaluated via ast.literal_eval
    - Malformed strings / NaN / other → empty list

    This replaces both safe_parse() in pipeline.py and fix_list() in
    feature_engineering.py — they were duplicates with different names.
    """
    if isinstance(val, list):
        return val
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return []
    if isinstance(val, str):
        # Normalize HTML entities before evaluation
        val = val.replace("&amp;", "&")
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (ValueError, SyntaxError):
            # Plain comma-separated string fallback
            stripped = val.strip()
            return [stripped] if stripped else []
    return []


def flatten_to_string(val) -> str:
    """
    Convert a list-like genre or tag value to a flat comma-separated string.
    Used by feature_engineering.py for the text-format columns.
    """
    lst = safe_parse(val)
    return ", ".join(lst) if lst else ""


# ── interaction normalization ─────────────────────────────────────────────────

def compute_log_playtime_interaction(
    playtime: float | np.ndarray,
    max_log_playtime: float,
) -> float | np.ndarray:
    """
    Convert raw playtime (minutes) into a normalized interaction score [0, 5].

    Formula (mirrors stage1_split_sides):
      log_pt   = log1p(playtime)
      score    = (log_pt / max_log_playtime) * 5

    The max_log_playtime must be the per-user maximum, computed across the user's
    full play history. Passing a single value is safe for the single-event case
    in nearline updates (use the stored user max, or fall back to log1p(playtime)).

    Args:
      playtime          : raw playtime in minutes (scalar or array)
      max_log_playtime  : the user's maximum log-playtime across all games
    """
    log_pt = np.log1p(np.asarray(playtime, dtype=np.float32))
    if max_log_playtime <= 0:
        return np.zeros_like(log_pt)
    return (log_pt / max_log_playtime) * 5.0


# ── user vector construction ───────────────────────────────────────────────────

def build_user_vector(
    user_history: pd.DataFrame,
    item_content_matrix: np.ndarray,
    item_vocab: dict[str, int],
) -> np.ndarray:
    """
    Build a single user's feature vector from their play history.

    This is the CRITICAL shared function. It produces the exact same mathematical
    output as _get_user_features() inside stage3_process_users. Training and the
    nearline updater must call this same function.

    Vector layout (322 dims by default for 20 genres + 300 tags):
      [0 : content_dim]    Interaction-weighted average of item content vectors.
      [content_dim]        Raw total interaction score (sum of all weights).
      [content_dim + 1]    Total number of distinct items played.

    Note: The last two dimensions are NOT normalized here. Call
    normalize_user_matrix() across ALL users to apply global max normalization
    before storing or passing to the model tower.

    Args:
      user_history         : DataFrame with columns ['item_name', 'interaction']
      item_content_matrix  : (num_items, content_dim) float32 array
      item_vocab           : dict mapping item_name -> row index in item_content_matrix

    Returns:
      np.ndarray of shape (content_dim + 2,), dtype float32
    """
    feat_dim = item_content_matrix.shape[1]
    zero_vec = np.zeros(feat_dim + 2, dtype=np.float32)

    valid = [
        (item_vocab[name], weight)
        for name, weight in zip(user_history["item_name"], user_history["interaction"])
        if name in item_vocab and weight > 0
    ]
    if not valid:
        return zero_vec

    indices, weights = zip(*valid)
    weights_arr = np.array(weights, dtype=np.float32)
    vecs = item_content_matrix[list(indices)]

    # Weighted average of content vectors, scaled by interaction strength
    history_vec = (vecs * weights_arr[:, None]).sum(axis=0) / (weights_arr.sum() + 1e-9)
    total_interaction = float(weights_arr.sum())
    num_games = float(len(valid))

    return np.concatenate([history_vec, [total_interaction, num_games]]).astype(np.float32)


def normalize_user_matrix(
    all_user_vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Apply global max normalization to the statistical tail of each user vector
    (the last two dimensions: total_interaction and num_games).

    This MUST be called on the full set of users to compute consistent normalization
    constants — the same constants that training used. For single-user nearline
    updates, either skip normalization (accepted approximation for online serving)
    or re-normalize against the stored population statistics.

    Args:
      all_user_vectors : dict mapping user_id -> raw (pre-normalization) vector

    Returns:
      A new dict with normalized vectors. The input dict is not mutated.
    """
    if not all_user_vectors:
        return {}

    user_ids = list(all_user_vectors.keys())
    matrix = np.stack([all_user_vectors[uid] for uid in user_ids])

    max_total = matrix[:, -2].max() + 1e-9
    max_ngames = matrix[:, -1].max() + 1e-9

    matrix[:, -2] /= max_total
    matrix[:, -1] /= max_ngames

    return {uid: matrix[i] for i, uid in enumerate(user_ids)}
