"""
Logging utility.

Provides a pre-configured logger used throughout the training pipeline.
"""

import logging
import sys


def _build_logger(name: str = "game_recommender") -> logging.Logger:
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    return _logger


logger = _build_logger()
