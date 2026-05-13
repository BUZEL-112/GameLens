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


class FeatureEngineeringService:
    """
    Service class for performing feature engineering on the cleaned dataset.
    """

    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
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

            # --- Step 2: List Sanitization Helper ---
            def fix_list(x):
                """
                Standardizes genre and tag entries into clean lists.
                Handles actual lists, string-represented lists, and malformed strings.
                """
                if isinstance(x, list):
                    return x
                if pd.isna(x):
                    return []
                if isinstance(x, str):
                    # Clean up HTML entities and try to evaluate as a Python list
                    x = x.replace("&amp;", "&")
                    try:
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        # Fallback for plain strings
                        return [x]
                return [str(x)]

            # --- Step 3: Process Genres and Tags ---
            # Convert raw list-like data into flat, comma-separated strings
            df["genres_1"] = (
                df["genres"]
                .apply(fix_list)
                .apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
            )
            df["tags_1"] = (
                df["tags"]
                .apply(fix_list)
                .apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
            )

            # Cleanup: Replace old columns with cleaned versions and remove raw title
            df.drop(columns=["genres", "tags"], inplace=True)
            df.rename(columns={"genres_1": "genres", "tags_1": "tags"}, inplace=True)

            if "title" in df.columns:
                df.drop(columns="title", inplace=True)

            # --- Step 4: Create 'item_text' for Content Features ---
            # Combines text features into a single searchable/embeddable field
            df["item_text"] = (
                (
                    df["genres"].fillna("").str.replace(",", " ")
                    + " "
                    + df["tags"].fillna("").str.replace(",", " ")
                )
                .str.lower()
                .str.strip()
            )
            # Filter out records with no descriptive text
            df = df[df["item_text"] != ""]
            
            # --- Step 5: Sampling and Persistence ---
            # Sub-sample to 10% for faster iteration during model development
            df = df.sample(frac=0.1)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(self.transformed_data_path), exist_ok=True)
            
            # Save the full processed dataset
            df.to_csv(self.transformed_data_path, index=False)
            self.logger.info(f"Transformed data saved to {self.transformed_data_path}")

            # --- Step 6: Train/Test Split ---
            # Divide into training and validation sets (80/20)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save individual splits
            train_df.to_csv(self.transformed_train_path, index=False)
            test_df.to_csv(self.transformed_test_path, index=False)

            self.logger.info(f"Train data saved to {self.transformed_train_path}")
            self.logger.info(f"Test data saved to {self.transformed_test_path}")
            self.logger.info("Feature engineering completed successfully.")
            
            return self.transformed_data_path

        except Exception as e:
            # Propagate error with custom context
            raise CustomException(e)
