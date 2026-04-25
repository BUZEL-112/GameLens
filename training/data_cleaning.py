"""
Data Cleaning and Merging Module

Processes raw gzipped data files (user-item interactions and Steam game metadata)
into a consolidated CSV ready for feature engineering.
"""

import os
import gzip
import ast

import pandas as pd
from pathlib import Path

from training.utils.utils import load_config
from training.utils.exception import CustomException
from training.utils.logger import logger


class CleanDataService:
    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
        try:
            self.config = load_config(config_path)
            self.logger = logger

            ingest_cfg = self.config.get("data_ingestion", {})
            clean_cfg = self.config.get("data_cleaning", {})

            self.raw_data_dir = clean_cfg.get("raw_data_dir", "data/raw")
            self.processed_dir = clean_cfg.get("root_dir", "data/processed")

            # Derive filenames from ingestion config URLs
            self.user_items_filename = ingest_cfg.get(
                "user_item_dataset_download_url", ""
            ).split("/")[-1]
            self.steam_games_filename = ingest_cfg.get(
                "steam_game_dataset_download_url", ""
            ).split("/")[-1]

            self.user_items_path = os.path.join(
                self.raw_data_dir, self.user_items_filename
            )
            self.steam_games_path = os.path.join(
                self.raw_data_dir, self.steam_games_filename
            )

            if not os.path.exists(self.user_items_path):
                raise FileNotFoundError(f"File {self.user_items_path} does not exist")
            if not os.path.exists(self.steam_games_path):
                raise FileNotFoundError(f"File {self.steam_games_path} does not exist")

            os.makedirs(self.processed_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e)

    def run(self) -> str:
        """
        Run cleaning pipeline. Returns path to the merged CSV.
        """
        try:
            self.logger.info("Starting data cleaning process...")

            output_path = os.path.join(
                self.processed_dir, "australian_users_items_merged.csv"
            )
            if os.path.exists(output_path):
                self.logger.info(
                    f"Cleaned data already exists at {output_path}. Skipping."
                )
                return output_path

            # --- Process User Items ---
            self.logger.info(f"Processing {self.user_items_filename}...")
            rows = []
            with gzip.open(self.user_items_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        user_data = ast.literal_eval(line.strip())
                        user_id = user_data["user_id"]
                        for item in user_data["items"]:
                            rows.append(
                                {
                                    "user_id": user_id,
                                    "item_id": item["item_id"],
                                    "playtime": item["playtime_forever"],
                                    "item_name": item["item_name"],
                                }
                            )
                    except Exception as e:
                        self.logger.warning(f"Error parsing line in user items: {e}")
                        continue

            df_users = pd.DataFrame(rows)
            self.logger.info(f"User items loaded. Shape: {df_users.shape}")

            # --- Process Steam Games ---
            self.logger.info(f"Processing {self.steam_games_filename}...")
            steam_data = []
            with gzip.open(self.steam_games_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        game_dict = ast.literal_eval(line.strip())
                        steam_data.append(game_dict)
                    except (ValueError, SyntaxError):
                        continue

            steam_df = pd.DataFrame(steam_data)

            cols_to_keep = ["id", "genres", "tags", "title"]
            existing_cols = [c for c in cols_to_keep if c in steam_df.columns]
            steam_df = steam_df[existing_cols]

            if "id" in steam_df.columns:
                steam_df = steam_df.rename(columns={"id": "item_id"})

            steam_df["item_id"] = steam_df["item_id"].astype(str)
            self.logger.info(f"Steam games loaded. Shape: {steam_df.shape}")

            # --- Merge ---
            self.logger.info("Merging datasets...")
            df_users["item_id"] = df_users["item_id"].astype(str)
            df_merged = df_users.merge(steam_df, on="item_id", how="left")

            df_merged.to_csv(output_path, index=False)
            self.logger.info(f"Data cleaning completed. Saved to {output_path}")
            return output_path

        except Exception as e:
            raise CustomException(e)
