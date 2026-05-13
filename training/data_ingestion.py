"""
Data Ingestion Module

This module handles the retrieval of raw datasets required for the training pipeline.
It follows these steps:
1. Initialization: Loads the project configuration (YAML) to identify source URLs and local storage paths.
2. Directory Setup: Ensures the destination directory for raw data exists on the local filesystem.
3. Availability Check: Iterates through the configured URLs and checks if the files are already present locally.
4. Data Acquisition: Downloads missing files from the remote URLs using a utility function.
5. Logging: Provides detailed status updates throughout the process for monitoring and debugging.
"""

import os
from pathlib import Path

# Import custom utilities for file handling, configuration loading, and error management
from training.utils.utils import download_file, load_config
from training.utils.exception import CustomException
from training.utils.logger import logger


class LoadDataService:
    """
    Service class responsible for managing the data ingestion workflow.
    """

    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
        """
        Initializes the service by loading configurations and setting up directories.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        try:
            # Load the main configuration file
            self.config = load_config(config_path)
            self.logger = logger

            # Extract data ingestion specific settings
            load_cfg = self.config.get("data_ingestion", {})

            # Define the source URLs for the datasets (with fallbacks to default Steam datasets)
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
            
            # Set the local directory where raw data will be stored
            self.raw_data_dir = load_cfg.get("raw_data_dir", "data/raw")

            # Create the directory if it doesn't already exist
            if self.raw_data_dir:
                os.makedirs(self.raw_data_dir, exist_ok=True)
            else:
                raise ValueError("raw_data_dir not found in configuration")

        except Exception as e:
            # Wrap any initialization errors in a CustomException
            raise CustomException(e)

    def run(self):
        """
        Executes the data ingestion process.
        """
        try:
            self.logger.info("Starting data ingestion process...")

            # Process each URL defined in the configuration
            for url in self.urls:
                self.logger.info(f"url={url}")
                
                # Derive the filename from the URL
                filename = url.split("/")[-1]
                file_path = os.path.join(self.raw_data_dir, filename)

                # Skip download if the file is already available locally
                if os.path.exists(file_path):
                    self.logger.info(f"File already exists: {file_path}")
                else:
                    self.logger.info(
                        f"File does not exist: {file_path}. Downloading..."
                    )
                    # Trigger the download utility
                    download_file(url, self.raw_data_dir)
                    self.logger.info(f"Successfully downloaded: {filename}")

            self.logger.info("Data ingestion completed.")

        except Exception as e:
            # Wrap any execution errors in a CustomException
            raise CustomException(e)
