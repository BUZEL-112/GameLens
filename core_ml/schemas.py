"""
core_ml.schemas — Shared data contracts for user and item vectors.

These dataclasses define the exact shape of every feature object that crosses
the boundary between training and serving. Both training/pipeline.py and
recommendation_api/services/nearline.py must agree on these shapes.

If the dimensionality or composition of a vector changes, update it here first
and let the type-checker surface every callsite that needs updating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UserVector:
    """
    Fixed-length 322-dimensional representation of a user's play history.

    Layout (matches stage3_process_users output):
      [0 : content_dim]      Interaction-weighted average of item content vectors.
                             content_dim = num_genres + num_tags (default 20 + 300 = 320).
      [content_dim]          Normalized total interaction score (volume of play).
      [content_dim + 1]      Normalized total number of distinct games played (diversity).

    The normalization of the last two dimensions is corpus-level (global max),
    so it must be applied across ALL users in the same batch — see
    core_ml.features.normalize_user_matrix().
    """

    user_id: str
    vec: np.ndarray  # shape: (322,), dtype: float32

    @property
    def dim(self) -> int:
        return self.vec.shape[0]


@dataclass
class ItemFeatures:
    """
    Content feature representation of a single item (game).

    Fields:
      item_name   — canonical string identifier (primary key in all lookups)
      item_idx    — integer index into item_vocab and item_content_matrix
      genres      — list of genre strings (parsed, not multi-hot encoded)
      tags        — list of tag strings (parsed, not multi-hot encoded)
    """

    item_name: str
    item_idx: int
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
