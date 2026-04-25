"""
Shared utilities used by the data pipeline services.
"""

import os
import yaml
import requests
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load a YAML configuration file and return it as a dictionary."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_dir: str) -> str:
    """
    Download a file from *url* into *dest_dir*.

    Returns the full path of the downloaded file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_dir, filename)

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_path
