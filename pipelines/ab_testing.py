"""
A/B Testing Framework

Manages experiment allocation, metric collection, and statistical
analysis for comparing model versions.

Usage:
    cd game_recommender
    python -m pipelines.ab_testing --action status
    python -m pipelines.ab_testing --action create --name "v2_hard_neg" --traffic 0.1
    python -m pipelines.ab_testing --action analyze --name "v2_hard_neg"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from training.utils.logger import logger


@dataclass
class Experiment:
    name: str
    control_model: str              # e.g. "v1.0"
    treatment_model: str            # e.g. "v2_hard_neg"
    traffic_fraction: float         # fraction of users routed to treatment
    status: str = "running"         # running | paused | concluded
    created_at: float = 0.0
    metrics: dict = None            # aggregated metrics per group

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.metrics is None:
            self.metrics = {"control": {}, "treatment": {}}


class ABTestingManager:
    """
    Manages experiment definitions and user routing.

    Experiments are stored as JSON files in a local directory.
    In production, this would be backed by a database or Redis.
    """

    def __init__(self, experiments_dir: str = "model_artifacts/experiments"):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.experiments_dir, f"{name}.json")

    def create_experiment(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_fraction: float = 0.1,
    ) -> Experiment:
        exp = Experiment(
            name=name,
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_fraction=traffic_fraction,
        )
        with open(self._path(name), "w") as f:
            json.dump(asdict(exp), f, indent=2)
        logger.info(
            f"Created experiment '{name}': "
            f"{control_model} vs {treatment_model} "
            f"({traffic_fraction * 100:.0f}% traffic)"
        )
        return exp

    def get_experiment(self, name: str) -> Optional[Experiment]:
        path = self._path(name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return Experiment(**data)

    def list_experiments(self) -> list[Experiment]:
        experiments = []
        for fname in os.listdir(self.experiments_dir):
            if fname.endswith(".json"):
                with open(os.path.join(self.experiments_dir, fname)) as f:
                    data = json.load(f)
                experiments.append(Experiment(**data))
        return experiments

    def route_user(self, user_id: str, experiment_name: str) -> str:
        """
        Deterministically assign a user to control or treatment group.

        Uses a hash of (user_id, experiment_name) so the assignment
        is stable across requests but varies per experiment.
        """
        exp = self.get_experiment(experiment_name)
        if exp is None or exp.status != "running":
            return "control"

        hash_input = f"{user_id}:{experiment_name}"
        hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_val % 1000) / 1000.0

        return "treatment" if bucket < exp.traffic_fraction else "control"

    def record_metric(
        self,
        experiment_name: str,
        group: str,
        metric_name: str,
        value: float,
    ):
        """Record a metric observation for a given group."""
        exp = self.get_experiment(experiment_name)
        if exp is None:
            return

        if metric_name not in exp.metrics[group]:
            exp.metrics[group][metric_name] = []
        exp.metrics[group][metric_name].append(value)

        with open(self._path(experiment_name), "w") as f:
            json.dump(asdict(exp), f, indent=2)

    def analyze(self, experiment_name: str) -> dict:
        """Compute summary statistics for each group."""
        exp = self.get_experiment(experiment_name)
        if exp is None:
            return {}

        import numpy as np

        summary = {}
        for group in ["control", "treatment"]:
            summary[group] = {}
            for metric, values in exp.metrics.get(group, {}).items():
                arr = np.array(values)
                summary[group][metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "count": len(values),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                }
        return summary

    def conclude(self, experiment_name: str, winner: str):
        """Mark an experiment as concluded with a winner."""
        exp = self.get_experiment(experiment_name)
        if exp is None:
            return
        exp.status = "concluded"
        with open(self._path(experiment_name), "w") as f:
            data = asdict(exp)
            data["winner"] = winner
            json.dump(data, f, indent=2)
        logger.info(f"Experiment '{experiment_name}' concluded — winner: {winner}")


def main():
    parser = argparse.ArgumentParser(description="A/B Testing management")
    parser.add_argument("--action", choices=["status", "create", "analyze"],
                        required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--control", default="v1.0")
    parser.add_argument("--treatment", default="v2.0")
    parser.add_argument("--traffic", type=float, default=0.1)
    args = parser.parse_args()

    mgr = ABTestingManager()

    if args.action == "status":
        experiments = mgr.list_experiments()
        if not experiments:
            print("No experiments found.")
        for exp in experiments:
            print(
                f"  {exp.name:<25} {exp.status:<12} "
                f"{exp.control_model} vs {exp.treatment_model} "
                f"({exp.traffic_fraction * 100:.0f}% traffic)"
            )

    elif args.action == "create":
        if not args.name:
            print("--name is required for create")
            return
        mgr.create_experiment(
            name=args.name,
            control_model=args.control,
            treatment_model=args.treatment,
            traffic_fraction=args.traffic,
        )

    elif args.action == "analyze":
        if not args.name:
            print("--name is required for analyze")
            return
        summary = mgr.analyze(args.name)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
