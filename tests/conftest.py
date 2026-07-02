"""
Shared pytest configuration and fixtures.

Fixtures defined here are available to all test modules without explicit imports.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Suppress verbose TensorFlow / Keras startup noise during unit test runs.
# The api and training unit tests mock away the model loading, but importing
# recommendation_api modules may still trigger TF registration messages.
# ---------------------------------------------------------------------------

def pytest_configure(config):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
