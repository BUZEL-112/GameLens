"""
Prefect Orchestrator for the Recommender Pipeline

Executes the dependency-based DAG:
Retention -> Validation -> Retraining -> Evaluation -> Deployment -> A/B Routing

Run with:
    python -m pipelines.orchestrator
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from training.utils.utils import load_config
from pipelines.data_retention import main as run_retention
from pipelines.data_validation import main as run_validation
from pipelines.retrain import main as run_retrain
from training.evaluate import main as run_evaluate
from pipelines.ab_testing import ABTestingManager


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="data_retention")
def task_data_retention(config_path: str):
    logger = get_run_logger()
    logger.info("Starting Data Retention task...")
    run_retention(config_path)

@task(name="data_validation")
def task_data_validation(config_path: str, wait_for=None):
    logger = get_run_logger()
    logger.info("Starting Data Validation task...")
    run_validation(config_path)

@task(name="model_retraining")
def task_model_retraining(config_path: str, wait_for=None):
    logger = get_run_logger()
    logger.info("Starting Model Retraining task...")
    run_retrain(config_path)

@task(name="offline_evaluation")
def task_offline_evaluation(config_path: str, wait_for=None):
    logger = get_run_logger()
    logger.info("Starting Offline Evaluation task...")
    run_evaluate(config_path)

@task(name="conditional_deployment")
def task_conditional_deployment(config_path: str, wait_for=None):
    logger = get_run_logger()
    logger.info("Starting Conditional Deployment task...")
    
    config = load_config(config_path)
    manifest_dir = config.get("orchestration", {}).get("manifest_dir", "model_artifacts/manifests")
    
    metrics_path = os.path.join(manifest_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Missing metrics manifest at {metrics_path}")
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    eval_cfg = config.get("evaluation", {})
    min_recall_20 = eval_cfg.get("min_recall_20", 0.0)
    min_ndcg_20 = eval_cfg.get("min_ndcg_20", 0.0)
    
    # ── Gate 1: Baseline Parity ───────────────────────────────────────────────
    if not metrics.get("baseline_gate_passed", False):
        raise RuntimeError("Deployment Gate Failed: Model did not pass the popularity baseline.")
        
    # ── Gate 2: Absolute Thresholds ───────────────────────────────────────────
    actual_recall_20 = metrics.get("recall_scores", {}).get("20", 0.0)
    actual_ndcg_20 = metrics.get("ndcg_scores", {}).get("20", 0.0)
    
    if actual_recall_20 < min_recall_20:
        raise RuntimeError(f"Deployment Gate Failed: Recall@20 ({actual_recall_20:.4f}) < Minimum ({min_recall_20})")
    
    if actual_ndcg_20 < min_ndcg_20:
        raise RuntimeError(f"Deployment Gate Failed: NDCG@20 ({actual_ndcg_20:.4f}) < Minimum ({min_ndcg_20})")
        
    logger.info("All Deployment Gates PASSED. Proceeding to push artifacts to Redis...")
    
    # ── Deployment Logic ──────────────────────────────────────────────────────
    try:
        from recommendation_api.core.config import settings
        from recommendation_api.core.feature_store import FeatureStore

        fs = FeatureStore(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
        )

        # Load training manifest for artifact paths
        t_manifest_path = os.path.join(manifest_dir, "training_manifest.json")
        with open(t_manifest_path, "r") as f:
            t_manifest = json.load(f)
            
        version_tag = t_manifest["version_tag"]

        # Load artifacts
        with open(t_manifest["artifacts_path"], "rb") as f:
            artifacts = pickle.load(f)
            
        item_embeddings = np.load(t_manifest["item_embeddings_path"])
        
        with open(t_manifest["similarity_table_path"], "rb") as f:
            similarity_table = pickle.load(f)

        # Load source data for populate call
        feat_cfg = config.get("feature_engineering", {})
        from training.pipeline import stage1_split_sides

        df = pd.read_csv(feat_cfg.get("transformed_data_path"))
        items_df, interactions_df = stage1_split_sides(df)

        # Compute popular items
        pop_totals = (
            interactions_df.groupby("item_name")["interaction"]
            .sum()
            .sort_values(ascending=False)
        )
        popular_items = [
            {"item_name": name, "score": float(pop_totals[name]), "source": "popularity"}
            for name in pop_totals.index[:50]
        ]

        fs.populate_from_artifacts(
            artifacts=artifacts,
            item_embeddings=item_embeddings,
            all_user_vectors=artifacts["all_user_vectors"],
            similarity_table=similarity_table,
            items_df=items_df,
            interactions_df=interactions_df,
            popular_items=popular_items,
        )
        fs.set_model_version(version_tag)
        fs.set_ready_sentinel(version_tag)
        fs.memory_report()
        
        logger.info(f"Deployment complete. New model version: {version_tag}")
        return version_tag

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

@task(name="ab_traffic_routing")
def task_ab_traffic_routing(new_version_tag: str, wait_for=None):
    logger = get_run_logger()
    logger.info("Starting A/B Traffic Routing task...")
    
    # In a real environment we might query Redis for the current active version, 
    # but here we'll use a placeholder or read the last concluded experiment.
    # For simplicity, we'll just set control = 'previous_version' and treatment = new_version_tag
    
    mgr = ABTestingManager()
    
    experiments = mgr.list_experiments()
    # Find the active experiment if there is one
    running_exps = [e for e in experiments if e.status == "running"]
    
    control_model = "v1.0"
    if running_exps:
        # If there's already an active experiment, conclude it with its treatment (or control)
        # and start a new one. Here we'll just use the old treatment as the new control.
        control_model = running_exps[0].treatment_model
        mgr.conclude(running_exps[0].name, winner=control_model)
        
    exp_name = f"exp_{new_version_tag}"
    
    mgr.create_experiment(
        name=exp_name,
        control_model=control_model,
        treatment_model=new_version_tag,
        traffic_fraction=0.10,  # 10% to the new model
    )
    logger.info(f"A/B experiment '{exp_name}' created routing 10% traffic to {new_version_tag}.")


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="Two-Tower Recommender Pipeline")
def recommender_pipeline(config_path: str = "configs/config.yaml"):
    logger = get_run_logger()
    logger.info("=== Starting Recommender Pipeline ===")
    
    # Task A
    t1 = task_data_retention.submit(config_path)
    
    # Task B depends on Task A
    t2 = task_data_validation.submit(config_path, wait_for=[t1])
    
    # Task C depends on Task B
    t3 = task_model_retraining.submit(config_path, wait_for=[t2])
    
    # Task D depends on Task C
    t4 = task_offline_evaluation.submit(config_path, wait_for=[t3])
    
    # Task E depends on Task D
    new_version_tag = task_conditional_deployment.submit(config_path, wait_for=[t4])
    
    # Task F depends on Task E
    task_ab_traffic_routing.submit(new_version_tag, wait_for=[new_version_tag])
    
    logger.info("Pipeline DAG submitted.")


if __name__ == "__main__":
    recommender_pipeline()
