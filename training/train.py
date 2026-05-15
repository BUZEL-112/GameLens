"""
Training Entry Point

This module orchestrates the end-to-end training pipeline for the Two-Tower recommendation system.
It covers everything from data acquisition to model artifact generation.

The pipeline follows these steps:
1. Setup: Loads configurations and initializes the output directory for model artifacts.
2. Data Preparation: Runs Ingestion, Cleaning, and Feature Engineering services.
3. Component Processing:
   - Stage 1-3: Separates items/users and builds vocabulary-based feature matrices.
   - Stage 4-5: Generates training pairs (positive and negative) and converts them to tensors.
4. Model Training: Builds the Dual-Tower architecture and executes the contrastive training loop.
5. Embedding Extraction: Uses the trained Item Tower to generate latent vectors for all games.
6. Indexing & Similarity: Builds a FAISS index for fast retrieval and pre-computes an item-to-item similarity table.
7. Persistence: Saves all models, weights, indexes, and metadata required for the serving API.

Usage:
    cd game_recommender
    python -m training.train
    python -m training.train --config configs/config.yaml
    python -m training.train --skip-download   # if data already on disk
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import faiss
import numpy as np
import pandas as pd

# Internal module imports for data handling and modeling
from training.utils.utils import load_config
from training.utils.logger import logger
from training.data_ingestion import LoadDataService
from training.data_cleaning import CleanDataService
from training.feature_engineering import FeatureEngineeringService
from training.models import build_user_tower, build_item_tower
from training.pipeline import (
    stage1_split_sides,
    stage2_process_items,
    stage3_process_users,
    stage4_build_training_pairs,
    stage5_assemble_tensors,
    stage6_train_loop,
)


def main(config_path: str = "configs/config.yaml", skip_download: bool = False, frac_dat: bool = True):
    """
    Main execution logic for the training pipeline.
    """
    t_start = time.perf_counter()
    
    # ── Phase 1: Configuration & Setup ────────────────────────────────────────
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    save_dir = config.get("serving", {}).get("artifacts_path", "model_artifacts")
    os.makedirs(save_dir, exist_ok=True)

    # Global Seeding for Reproducibility
    random_seed = train_cfg.get("random_seed", 42)
    import random
    import tensorflow as tf
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    logger.info(f"Global random seed set to {random_seed}")

    # ── Phase 2: Data Preparation ─────────────────────────────────────────────
    # These services ensure the local 'data/' directory is populated with model-ready CSVs
    if not skip_download:
        logger.info("=== Data Ingestion ===")
        LoadDataService(config_path).run()

    logger.info("=== Data Cleaning ===")
    CleanDataService(config_path).run()

    logger.info("=== Feature Engineering ===")
    _, sampling_stats = FeatureEngineeringService(config_path, frac_dat=frac_dat).run()

    # ── Phase 3: Load Transformed Data ────────────────────────────────────────
    feat_cfg = config.get("feature_engineering", {})
    transformed_path = feat_cfg.get("transformed_data_path", "data/processed/transformed_df.csv")
    logger.info(f"Loading transformed data from {transformed_path}")
    df = pd.read_csv(transformed_path)

    # ── Phase 4: Pipeline Stages 1-6 ─────────────────────────────────────────
    # Stage 1: Separate the consolidated dataframe into unique items and user interactions
    logger.info("=== Stage 1: Splitting sides ===")
    items_df, interactions_df = stage1_split_sides(df)

    # Stage 2: Build item vocabulary and content matrix (genres/tags)
    logger.info("=== Stage 2: Processing items ===")
    item_data = stage2_process_items(
        items_df,
        num_genres=train_cfg.get("num_genres", 20),
        num_tags=train_cfg.get("num_tags", 300),
    )

    # Stage 3: Aggregate user history into fixed-length feature vectors
    logger.info("=== Stage 3: Processing users ===")
    all_user_vectors = stage3_process_users(
        interactions_df,
        item_data["item_content_matrix"],
        item_data["item_vocab"],
    )

    # Stage 4: Sample positive interactions and generate synthetic negative samples
    logger.info("=== Stage 4: Building training pairs ===")
    train_samples = stage4_build_training_pairs(
        interactions_df,
        item_data["item_vocab"],
        item_data["idx_to_name"],
        item_data["item_content_matrix"],
        item_data["num_items"],
        neg_per_pos_hard=train_cfg.get("neg_per_pos_hard", 2),
        neg_per_pos_random=train_cfg.get("neg_per_pos_random", 2),
        min_interaction=train_cfg.get("min_interaction", 0.01),
    )

    # Resolve architectural dimensions
    user_feat_dim = next(iter(all_user_vectors.values())).shape[0]
    embedding_dim = train_cfg.get("embedding_dim", 128)
    item_content_dim = item_data["item_content_matrix"].shape[1]

    # Stage 5: Convert Python lists/dicts into high-performance TensorFlow tensors
    logger.info("=== Stage 5: Assembling tensors ===")
    tensors = stage5_assemble_tensors(
        train_samples,
        all_user_vectors,
        item_data["item_vocab"],
        item_data["item_content_matrix"],
        user_feat_dim=user_feat_dim,
        n_train=train_cfg.get("n_train_samples"),
    )

    # ── Phase 5: Model Construction & Training ────────────────────────────────
    logger.info("=== Building towers ===")
    u_tower = build_user_tower(user_feat_dim, embedding_dim)
    i_tower = build_item_tower(
        item_data["num_items"],
        content_dim=item_content_dim,
        id_emb_dim=train_cfg.get("item_id_emb_dim", 32),
        output_dim=embedding_dim,
    )
    u_tower.summary(print_fn=logger.info)
    i_tower.summary(print_fn=logger.info)

    # Stage 6: Execute the gradient descent loop with Contrastive Loss
    logger.info("=== Stage 6: Training ===")
    history = stage6_train_loop(
        u_tower, i_tower, tensors,
        epochs=train_cfg.get("epochs", 100),
        batch_size=train_cfg.get("batch_size", 512),
        temperature=train_cfg.get("temperature", 0.1),
        lr_initial=train_cfg.get("lr_initial", 1e-3),
        lr_floor=train_cfg.get("lr_floor", 1e-5),
    )

    # ── Phase 6: Embedding Extraction & FAISS Indexing ───────────────────────
    # Generate embeddings for the entire item catalog to enable fast retrieval
    logger.info("=== Extracting item embeddings ===")
    num_items = item_data["num_items"]
    all_item_ids = np.arange(num_items, dtype=np.int32).reshape(-1, 1)
    all_item_content = item_data["item_content_matrix"]

    item_embeddings = []
    EMB_BATCH = 1024
    for start in range(0, num_items, EMB_BATCH):
        end = min(start + EMB_BATCH, num_items)
        emb = i_tower(
            [all_item_ids[start:end], all_item_content[start:end]], training=False
        ).numpy()
        item_embeddings.append(emb)
    item_embeddings = np.vstack(item_embeddings)

    logger.info(f"Item embeddings shape: {item_embeddings.shape}")

    # Build FAISS Index (Inner Product for Cosine Similarity since vectors are normalized)
    d = embedding_dim
    item_emb_f32 = np.ascontiguousarray(item_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(d)
    index.add(item_emb_f32)
    logger.info(f"FAISS index built — {index.ntotal:,} items")

    # ── Phase 7: Similarity Table Pre-computation ────────────────────────────
    # Generate a static lookup table for "Similar Games" features
    logger.info("Pre-computing similarity table...")
    idx_to_name = item_data["idx_to_name"]
    similarity_table = {}
    for start in range(0, num_items, 512):
        end = min(start + 512, num_items)
        batch_embs = item_emb_f32[start:end]
        scores_batch, indices_batch = index.search(batch_embs, 11)
        for local_i, (idxs, scs) in enumerate(zip(indices_batch, scores_batch)):
            item_idx = start + local_i
            item_name = idx_to_name.get(item_idx)
            if item_name is None:
                continue
            # Store top 10 neighbors (excluding self)
            similarity_table[item_name] = [
                {"item_name": idx_to_name[int(i)], "score": float(s)}
                for i, s in zip(idxs, scs)
                if int(i) in idx_to_name and int(i) != item_idx
            ][:10]

    # ── Phase 8: Artifact Persistence ────────────────────────────────────────
    # Save all files required by the Recommendation API
    logger.info(f"=== Saving artifacts to {save_dir}/ ===")

    u_tower.save(f"{save_dir}/u_tower.keras")
    i_tower.save(f"{save_dir}/i_tower.keras")
    u_tower.save_weights(f"{save_dir}/u_tower.weights.h5")
    i_tower.save_weights(f"{save_dir}/i_tower.weights.h5")
    logger.info("  Tower models saved")

    np.save(f"{save_dir}/item_embeddings.npy", item_embeddings)
    logger.info(f"  Item embeddings saved {item_embeddings.shape}")

    faiss.write_index(index, f"{save_dir}/item_index.faiss")
    logger.info(f"  FAISS index saved ({index.ntotal:,} items)")

    # Metadata and Vocabularies
    artifacts = {
        "item_vocab": item_data["item_vocab"],
        "genre_vocab": item_data["genre_vocab"],
        "tag_vocab": item_data["tag_vocab"],
        "idx_to_name": idx_to_name,
        "all_user_vectors": all_user_vectors,
        "num_items": num_items,
        "USER_FEAT_DIM": user_feat_dim,
        "EMBEDDING_DIM": embedding_dim,
        "ITEM_CONTENT_DIM": item_content_dim,
    }
    with open(f"{save_dir}/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("  Vocabularies + user vectors saved")

    with open(f"{save_dir}/similarity_table.pkl", "wb") as f:
        pickle.dump(similarity_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  Similarity table saved ({len(similarity_table):,} items)")

    # Training telemetry
    with open(f"{save_dir}/training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    # Output a manifest of all saved files
    total_bytes = 0
    import hashlib
    import json
    from datetime import datetime

    def compute_md5(file_path):
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    version_tag = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"\n{save_dir}/")
    checksums = {}
    for fname in sorted(os.listdir(save_dir)):
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_bytes += size
            checksums[fname] = compute_md5(fpath)
            logger.info(f"  {fname:<35}  {size / 1e6:>7.2f} MB")
    logger.info(f"  {'TOTAL':<35}  {total_bytes / 1e6:>7.2f} MB")

    # ── Write Training Manifest ───────────────────────────────────────────────
    manifest = {
        "timestamp": time.time(),
        "version_tag": version_tag,
        "sampling": {
            "random_seed": random_seed,
            "max_interactions_per_user": train_cfg.get("max_interactions_per_user"),
            "sample_fraction": train_cfg.get("sample_fraction"),
            "note": "Time-aware sampling (.tail(N)) relies on inherent chronological order."
        },
        "row_counts": sampling_stats,
        "item_embeddings_path": f"{save_dir}/item_embeddings.npy",
        "artifacts_path": f"{save_dir}/artifacts.pkl",
        "similarity_table_path": f"{save_dir}/similarity_table.pkl",
        "checksums": checksums
    }

    manifest_dir = config.get("orchestration", {}).get("manifest_dir", "model_artifacts/manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "training_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Training manifest written for version: {version_tag}")

    elapsed = time.perf_counter() - t_start
    logger.info(f"\nTraining pipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Two-Tower recommender")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download (assumes raw data already exists)",
    )
    parser.add_argument(
        "--disable-sampling", action="store_false", dest="frac_dat",
        help="Disable 10% data sampling (use full dataset)",
    )
    args = parser.parse_args()
    main(config_path=args.config, skip_download=args.skip_download, frac_dat=args.frac_dat)
