"""
Feature Engineering Module

This module transforms cleaned data into a format suitable for training recommendation models.
It follows these steps:
1. Initialization: Resolves input/output paths from the configuration file.
2. Data Loading: Imports the cleaned dataset containing user interactions and game metadata.
3. Implicit Rating Calculation: Converts 'playtime' into a 'rating' using a logarithmic scale (log1p).
4. Data Sanitization: Uses a helper function to fix inconsistent list formats in genres and tags.
5. Text Formatting: Flattens genre and tag lists into clean, searchable strings.
6. Feature Synthesis: Creates an 'item_text' column by merging genres and tags for content-based analysis.
7. Sampling: Reduces dataset size (10% sample) to optimize training performance.
8. Data Splitting & Export: Splits the data into 80/20 train/test sets and saves all transformed files.
Note: By default, this service samples only 10% of the data to reduce training time for demonstrations. To use the full dataset, set 'frac_dat=False' in the class constructor.
"""

import os
import ast

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Custom error handling and utility imports
from training.utils.exception import CustomException
from training.utils.logger import logger
from training.utils.utils import load_config

# Shared feature functions — same logic used by the nearline API at serving time
from core_ml.features import safe_parse, flatten_to_string


class FeatureEngineeringService:
    """
    Service class for performing feature engineering on the cleaned dataset.
    """

    def __init__(self, config_path: str = str(Path("configs/config.yaml")), frac_dat:bool= True):
        """
        Initializes the service by loading paths for input data and output artifacts.
        """
        try:
            self.config = load_config(config_path)
            self.logger = logger

            # Retrieve feature engineering specific configurations
            feat_cfg = self.config.get("feature_engineering", {})
            self.root_dir = feat_cfg.get("root_dir")
            self.cleaned_data_path = feat_cfg.get("cleaned_data_path")
            self.transformed_train_path = feat_cfg.get("transformed_train_path")
            self.transformed_test_path = feat_cfg.get("transformed_test_path")
            self.transformed_data_path = feat_cfg.get("transformed_data_path")

        except Exception as e:
            raise CustomException(e)

    def run(self) -> str:
        """
        Executes the feature engineering pipeline.

        Returns:
            str: The path to the full transformed CSV file.
        """
        try:
            self.logger.info("Starting feature engineering process...")

            # Validate input availability
            if not os.path.exists(self.cleaned_data_path):
                raise FileNotFoundError(
                    f"Cleaned data file not found: {self.cleaned_data_path}"
                )

            # Load the cleaned dataset
            df = pd.read_csv(self.cleaned_data_path)
            self.logger.info(f"Loaded data with shape: {df.shape}")

            # --- Step 1: Create Implicit Ratings ---
            # Use log(1 + playtime) to dampen the effect of outliers while maintaining relative ranking
            df["playtime"] = df["playtime"].fillna(0).astype(int)
            df["rating"] = np.log1p(df["playtime"])

            # --- Step 2: List Sanitization and Text Formatting ---
            # Use shared safe_parse (same function used by the serving API).
            # fix_list() was a local duplicate — it is now removed.
            df["genres_1"] = df["genres"].apply(flatten_to_string)
            df["tags_1"] = df["tags"].apply(flatten_to_string)

            # Cleanup: Replace old columns with cleaned versions and remove raw title
            df.drop(columns=["genres", "tags"], inplace=True)
            df.rename(columns={"genres_1": "genres", "tags_1": "tags"}, inplace=True)

            if "title" in df.columns:
                df.drop(columns="title", inplace=True)


            # --- Step : Sampling and Persistence ---
            # Retrieve training configurations for deterministic sampling
            train_cfg = self.config.get("training", {})
            random_seed = train_cfg.get("random_seed", 42)
            max_interactions = train_cfg.get("max_interactions_per_user", 20)
            sample_fraction = train_cfg.get("sample_fraction", 0.1)

            raw_rows = len(df)

            # Time-aware sampling proxy: keep only the most recent N interactions per user
            # This relies on the chronological ordering of the input dataset.
            self.logger.info(f"Applying time-aware windowing: keeping latest {max_interactions} interactions per user.")
            df = df.groupby("user_id").tail(max_interactions).reset_index(drop=True)

            after_tail_rows = len(df)

            # Sub-sample for faster iteration during model development
            if frac_dat:
                self.logger.info(f"Randomly sampling {sample_fraction * 100}% of data with seed {random_seed}.")
                df = df.sample(frac=sample_fraction, random_state=random_seed)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(self.transformed_data_path), exist_ok=True)
            
            # Save the full processed dataset
            df.to_csv(self.transformed_data_path, index=False)
            self.logger.info(f"Transformed data saved to {self.transformed_data_path}")

            # --- Step : Train/Test Split ---
            # Divide into training and validation sets (80/20)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_seed)
            
            # Save individual splits
            train_df.to_csv(self.transformed_train_path, index=False)
            test_df.to_csv(self.transformed_test_path, index=False)

            self.logger.info(f"Train data saved to {self.transformed_train_path}")
            self.logger.info(f"Test data saved to {self.transformed_test_path}")
            self.logger.info("Feature engineering completed successfully.")
            
            stats = {
                "raw_rows": raw_rows,
                "after_tail_rows": after_tail_rows,
                "after_sampling_rows": len(df),
                "train_split_size": len(train_df),
                "test_split_size": len(test_df)
            }
            return self.transformed_data_path, stats

        except Exception as e:
            # Propagate error with custom context
            raise CustomException(e)
