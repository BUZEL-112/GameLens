"""
Two-Tower Model Definitions

This module defines the architectural components for the Dual-Tower retrieval model.
The towers project users and items into a shared embedding space to facilitate fast similarity search.

The construction follows these steps:
1. Dimension Definitions: Sets constants for input features (user history, game content) and the latent embedding space.
2. User Tower Construction:
   - Stabilizes wide-ranging input features using Batch Normalization.
   - Passes features through a series of Dense layers with Dropout for non-linear feature extraction.
   - Projects the output to the shared embedding dimension with Unit Normalization (L2) for cosine similarity.
3. Item Tower Construction:
   - Processes dual inputs: categorical Item IDs and numerical content vectors (genres/tags).
   - Learns a unique latent representation for every game via an Embedding layer.
   - Merges ID and content representations before passing through a symmetrical DNN structure.
   - Finalizes with L2-normalization to ensure embeddings lie on a hypersphere.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ── Default dimension constants ──────────────────────────────────────────────
# These constants define the shape of the joint embedding space and input layers
EMBEDDING_DIM = 128       # Output space dimensionality (shared by both towers)
USER_FEAT_DIM = 322       # Input: 320 history-based features + 2 statistical features
ITEM_CONTENT_DIM = 320    # Input: 20 genre-based flags + 300 tag-based flags
ITEM_ID_EMB_DIM = 32      # Dimensionality of the learned per-item ID embedding


def build_user_tower(
    input_dim: int = USER_FEAT_DIM,
    output_dim: int = EMBEDDING_DIM,
) -> Model:
    """
    Constructs the User Tower architecture.

    Architecture: 322-dim Input -> BatchNorm -> Dense(256) -> Dropout -> Dense(128) -> L2 Norm.
    
    Args:
        input_dim (int): Size of the user feature vector.
        output_dim (int): Final embedding size.

    Returns:
        Model: A Keras Model representing the user encoder.
    """
    # Input layer for user features
    inp = layers.Input(shape=(input_dim,), name="user_input")
    
    # BatchNorm is critical here to prevent features with high variance from dominating the gradient
    x = layers.BatchNormalization()(inp)
    
    # Deep layers for learning complex user preferences
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)  # Regularization to prevent overfitting to specific users
    x = layers.Dense(128, activation="relu")(x)
    
    # Linear projection to the shared space
    x = layers.Dense(output_dim, activation=None)(x)
    
    # Unit normalization ensures that dot products correspond to cosine similarity
    out = layers.UnitNormalization(axis=1, name="user_embedding")(x)
    
    return Model(inp, out, name="user_tower")


def build_item_tower(
    n_items: int,
    content_dim: int = ITEM_CONTENT_DIM,
    id_emb_dim: int = ITEM_ID_EMB_DIM,
    output_dim: int = EMBEDDING_DIM,
) -> Model:
    """
    Constructs the Item Tower architecture.

    Architecture: (Item_ID + Content) -> Concatenation -> Dense(256) -> Dropout -> Dense(128) -> L2 Norm.

    Args:
        n_items (int): Vocabulary size for the item embedding layer.
        content_dim (int): Size of the game content vector.
        id_emb_dim (int): Size of the latent ID embedding.
        output_dim (int): Final embedding size.

    Returns:
        Model: A Keras Model representing the item encoder.
    """
    # Multi-input architecture: Categorical ID and Numerical Content
    id_inp = layers.Input(shape=(1,), name="item_id_input")
    content_inp = layers.Input(shape=(content_dim,), name="item_content_input")

    # Learnable ID representation captures specific item biases/identities
    id_emb = layers.Embedding(n_items, id_emb_dim, name="item_id_emb")(id_inp)
    id_emb = layers.Flatten()(id_emb)

    # Merge content features (genres/tags) with the learned ID representation
    x = layers.Concatenate()([id_emb, content_inp])
    
    # Symmetric processing to the User Tower
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    
    # Final projection to shared space
    x = layers.Dense(output_dim, activation=None)(x)
    
    # L2-Normalization for cosine similarity search support
    out = layers.UnitNormalization(axis=1, name="item_embedding")(x)

    return Model([id_inp, content_inp], out, name="item_tower")
