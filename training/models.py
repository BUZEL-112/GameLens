"""
Two-Tower Model Definitions

Defines the User Tower and Item Tower architectures used in the
retrieval model.

User Tower: 322-dim input -> 128-dim L2-normalised embedding
Item Tower: (item_id + 320-dim content) -> 128-dim L2-normalised embedding
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ── Default dimension constants ──────────────────────────────────────────────
EMBEDDING_DIM = 128       # output space both towers project into
USER_FEAT_DIM = 322       # 320 history + 2 stats
ITEM_CONTENT_DIM = 320    # genres(20) + tags(300)
ITEM_ID_EMB_DIM = 32      # learned per-item ID embedding


def build_user_tower(
    input_dim: int = USER_FEAT_DIM,
    output_dim: int = EMBEDDING_DIM,
) -> Model:
    """
    Build the User Tower.

    322-dim -> 128-dim L2-normalised embedding.

    BatchNorm first so the wide dynamic range of the history vector
    does not saturate the first Dense layer.
    """
    inp = layers.Input(shape=(input_dim,), name="user_input")
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(output_dim, activation=None)(x)
    out = layers.UnitNormalization(axis=1, name="user_embedding")(x)
    return Model(inp, out, name="user_tower")


def build_item_tower(
    n_items: int,
    content_dim: int = ITEM_CONTENT_DIM,
    id_emb_dim: int = ITEM_ID_EMB_DIM,
    output_dim: int = EMBEDDING_DIM,
) -> Model:
    """
    Build the Item Tower.

    Two inputs:
      item_id  : integer tensor (batch, 1)  -> Embedding -> 32-dim
      content  : float tensor  (batch, 320) -> passed through
    Concatenated -> 352-dim -> tower -> 128-dim L2-normalised embedding.
    """
    id_inp = layers.Input(shape=(1,), name="item_id_input")
    content_inp = layers.Input(shape=(content_dim,), name="item_content_input")

    # Learnable ID representation
    id_emb = layers.Embedding(n_items, id_emb_dim, name="item_id_emb")(id_inp)
    id_emb = layers.Flatten()(id_emb)

    # Merge content + ID
    x = layers.Concatenate()([id_emb, content_inp])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(output_dim, activation=None)(x)
    out = layers.UnitNormalization(axis=1, name="item_embedding")(x)

    return Model([id_inp, content_inp], out, name="item_tower")
