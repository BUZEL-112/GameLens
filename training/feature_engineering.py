"""
Feature Engineering Module

Transforms cleaned data into model-ready format: creates implicit ratings,
processes genre/tag text, builds item_text column, and splits train/test.
"""

import os
import ast

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from training.utils.exception import CustomException
from training.utils.logger import logger
from training.utils.utils import load_config


class FeatureEngineeringService:
    """
    Service class for performing feature engineering on the cleaned dataset.
    """

    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
        try:
            self.config = load_config(config_path)
            self.logger = logger

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
        Execute the feature engineering pipeline.

        Returns the path to the full transformed CSV.
        """
        try:
            self.logger.info("Starting feature engineering process...")

            if not os.path.exists(self.cleaned_data_path):
                raise FileNotFoundError(
                    f"Cleaned data file not found: {self.cleaned_data_path}"
                )

            df = pd.read_csv(self.cleaned_data_path)
            self.logger.info(f"Loaded data with shape: {df.shape}")

            # Rating: log(1+playtime)
            df["playtime"] = df["playtime"].fillna(0).astype(int)
            df["rating"] = np.log1p(df["playtime"])

            def fix_list(x):
                if isinstance(x, list):
                    return x
                if pd.isna(x):
                    return []
                if isinstance(x, str):
                    x = x.replace("&amp;", "&")
                    try:
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        return [x]
                return [str(x)]

            # Process genres and tags
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

            df.drop(columns=["genres", "tags"], inplace=True)
            df.rename(columns={"genres_1": "genres", "tags_1": "tags"}, inplace=True)

            if "title" in df.columns:
                df.drop(columns="title", inplace=True)

            # Create item_text for content-based features
            df["item_text"] = (
                (
                    df["genres"].fillna("").str.replace(",", " ")
                    + " "
                    + df["tags"].fillna("").str.replace(",", " ")
                )
                .str.lower()
                .str.strip()
            )
            df = df[df["item_text"] != ""]

            os.makedirs(os.path.dirname(self.transformed_data_path), exist_ok=True)
            df.to_csv(self.transformed_data_path, index=False)
            self.logger.info(f"Transformed data saved to {self.transformed_data_path}")

            # Train/Test Split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.transformed_train_path, index=False)
            test_df.to_csv(self.transformed_test_path, index=False)

            self.logger.info(f"Train data saved to {self.transformed_train_path}")
            self.logger.info(f"Test data saved to {self.transformed_test_path}")
            self.logger.info("Feature engineering completed successfully.")
            return self.transformed_data_path

        except Exception as e:
            raise CustomException(e)
