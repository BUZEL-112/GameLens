"""
Data Ingestion Module

Ensures that the necessary raw datasets are available locally for downstream
processing. Downloads from configured URLs if not already present.
"""

import os
from pathlib import Path

from training.utils.utils import download_file, load_config
from training.utils.exception import CustomException
from training.utils.logger import logger


class LoadDataService:
    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
        try:
            self.config = load_config(config_path)
            self.logger = logger

            load_cfg = self.config.get("data_ingestion", {})

            self.urls = [
                load_cfg.get(
                    "user_item_dataset_download_url",
                    "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz",
                ),
                load_cfg.get(
                    "steam_game_dataset_download_url",
                    "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
                ),
            ]
            self.raw_data_dir = load_cfg.get("raw_data_dir", "data/raw")

            if self.raw_data_dir:
                os.makedirs(self.raw_data_dir, exist_ok=True)
            else:
                raise ValueError("raw_data_dir not found in configuration")

        except Exception as e:
            raise CustomException(e)

    def run(self):
        try:
            self.logger.info("Starting data ingestion process...")

            for url in self.urls:
                self.logger.info(f"url={url}")
                filename = url.split("/")[-1]
                file_path = os.path.join(self.raw_data_dir, filename)

                if os.path.exists(file_path):
                    self.logger.info(f"File already exists: {file_path}")
                else:
                    self.logger.info(
                        f"File does not exist: {file_path}. Downloading..."
                    )
                    download_file(url, self.raw_data_dir)
                    self.logger.info(f"Successfully downloaded: {filename}")

            self.logger.info("Data ingestion completed.")

        except Exception as e:
            raise CustomException(e)
