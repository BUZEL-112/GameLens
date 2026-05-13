"""
Data Cleaning and Merging Module

This module processes raw gzipped data files (user-item interactions and Steam game metadata)
into a consolidated CSV ready for feature engineering.

The pipeline follows these steps:
1. Initialization: Loads configuration and sets up paths for raw input and processed output.
2. Deduplication Check: Skips processing if the final merged file already exists.
3. User Item Extraction: Unzips and parses user interaction data, flattening nested lists of items.
4. Game Metadata Extraction: Unzips and parses Steam game metadata using literal evaluation.
5. Metadata Refinement: Selects key columns (id, genres, tags, title) and standardizes IDs to strings.
6. Dataset Merging: Performs a left join to combine user interactions with game metadata.
7. Data Export: Saves the consolidated dataframe to a CSV file.
"""

import os
import gzip
import ast

import pandas as pd
from pathlib import Path

# Custom utility and error handling imports
from training.utils.utils import load_config
from training.utils.exception import CustomException
from training.utils.logger import logger


class CleanDataService:
    """
    Service class to handle the cleaning and merging of raw Steam datasets.
    """

    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
        """
        Sets up the cleaning service by resolving paths from configuration.
        """
        try:
            self.config = load_config(config_path)
            self.logger = logger

            # Extract configuration segments
            ingest_cfg = self.config.get("data_ingestion", {})
            clean_cfg = self.config.get("data_cleaning", {})

            # Define directory locations
            self.raw_data_dir = clean_cfg.get("raw_data_dir", "data/raw")
            self.processed_dir = clean_cfg.get("root_dir", "data/processed")

            # Derive filenames from ingestion config URLs to ensure consistency
            self.user_items_filename = ingest_cfg.get(
                "user_item_dataset_download_url", ""
            ).split("/")[-1]
            self.steam_games_filename = ingest_cfg.get(
                "steam_game_dataset_download_url", ""
            ).split("/")[-1]

            # Construct full file paths
            self.user_items_path = os.path.join(
                self.raw_data_dir, self.user_items_filename
            )
            self.steam_games_path = os.path.join(
                self.raw_data_dir, self.steam_games_filename
            )

            # Validate that raw data files are present before proceeding
            if not os.path.exists(self.user_items_path):
                raise FileNotFoundError(f"File {self.user_items_path} does not exist")
            if not os.path.exists(self.steam_games_path):
                raise FileNotFoundError(f"File {self.steam_games_path} does not exist")

            # Ensure the processed data directory exists
            os.makedirs(self.processed_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e)

    def run(self) -> str:
        """
        Executes the cleaning pipeline. Returns the path to the resulting merged CSV of 
        users interaction and games information data.
        """
        try:
            self.logger.info("Starting data cleaning process...")

            # Define the output file path
            output_path = os.path.join(
                self.processed_dir, "australian_users_items_merged.csv"
            )
            
            # Optimization: Skip processing if output already exists
            if os.path.exists(output_path):
                self.logger.info(
                    f"Cleaned data already exists at {output_path}. Skipping."
                )
                return output_path

            # --- Step 1: Process User Items (Flatten nested structure) ---
            self.logger.info(f"Processing {self.user_items_filename}...")
            rows = []
            with gzip.open(self.user_items_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Use ast.literal_eval for safe evaluation of Python-like strings
                        user_data = ast.literal_eval(line.strip())
                        user_id = user_data["user_id"]
                        
                        # Flatten the 'items' list inside each user record
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

            # --- Step 2: Process Steam Games (Metadata) ---
            self.logger.info(f"Processing {self.steam_games_filename}...")
            steam_data = []
            with gzip.open(self.steam_games_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Parse each line of the game metadata
                        game_dict = ast.literal_eval(line.strip())
                        steam_data.append(game_dict)
                    except (ValueError, SyntaxError):
                        # Skip malformed lines
                        continue

            steam_df = pd.DataFrame(steam_data)

            # Filter for specific columns relevant to the recommendation model
            cols_to_keep = ["id", "genres", "tags", "title"]
            existing_cols = [c for c in cols_to_keep if c in steam_df.columns]
            steam_df = steam_df[existing_cols]

            # Standardize the ID column name for merging
            if "id" in steam_df.columns:
                steam_df = steam_df.rename(columns={"id": "item_id"})

            # Ensure ID is a string to match the user interaction data
            steam_df["item_id"] = steam_df["item_id"].astype(str)
            self.logger.info(f"Steam games loaded. Shape: {steam_df.shape}")

            # --- Step 3: Merge User Interactions and Game Metadata ---
            self.logger.info("Merging datasets...")
            df_users["item_id"] = df_users["item_id"].astype(str)
            
            # Perform left merge to keep all user interaction records
            df_merged = df_users.merge(steam_df, on="item_id", how="left")

            # Save the final consolidated dataset
            df_merged.to_csv(output_path, index=False)
            self.logger.info(f"Data cleaning completed. Saved to {output_path}")
            
            return output_path

        except Exception as e:
            # Re-wrap any errors in a CustomException for consistent error propagation
            raise CustomException(e)
