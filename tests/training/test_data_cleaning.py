"""
Unit tests for training.data_cleaning.CleanDataService

Tests cover the pure data-transformation logic in isolation:
- _extract_user_rows: flatten nested user-item structures from gzip input
- _extract_steam_rows: parse game metadata and select relevant columns
- _merge: left-join behaviour (all user rows kept, metadata attached where available)
- run() skips work when output file already exists
- run() raises CleanDataService when raw files are missing

No real gzip files or network calls required -- all I/O is replaced with
in-memory helpers via unittest.mock and pytest's tmp_path fixture.

Run with:
    pytest tests/training/test_data_cleaning.py -v
"""

from __future__ import annotations

import ast
import gzip
import io
import json
import os
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest

from training.utils.exception import CustomException


# ---------------------------------------------------------------------------
# Helpers: build synthetic gzip content
# ---------------------------------------------------------------------------

def _make_user_items_gz(records: list[dict]) -> bytes:
    """
    Encode a list of user-item dicts into the same gzip line-by-line format
    used by the real australian_users_items.json.gz dataset.
    Each record is written as a Python literal string (ast.literal_eval readable).
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for rec in records:
            line = repr(rec) + "\n"
            gz.write(line.encode("utf-8"))
    return buf.getvalue()


def _make_steam_games_gz(records: list[dict]) -> bytes:
    """
    Encode a list of game-metadata dicts into the gzip line-by-line format
    used by steam_games.json.gz.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for rec in records:
            line = repr(rec) + "\n"
            gz.write(line.encode("utf-8"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Minimal config that CleanDataService.__init__ will consume
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "data_ingestion": {
        "user_item_dataset_download_url": (
            "https://example.com/australian_users_items.json.gz"
        ),
        "steam_game_dataset_download_url": (
            "https://example.com/steam_games.json.gz"
        ),
    },
    "data_cleaning": {
        "raw_data_dir": "",   # filled in by each test via tmp_path
        "root_dir": "",       # filled in by each test via tmp_path
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    return tmp_path / "raw"


@pytest.fixture()
def processed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "processed"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def user_items_bytes() -> bytes:
    records = [
        {
            "user_id": "user_001",
            "items": [
                {"item_id": "10", "item_name": "Counter-Strike", "playtime_forever": 200},
                {"item_id": "20", "item_name": "Portal", "playtime_forever": 50},
            ],
        },
        {
            "user_id": "user_002",
            "items": [
                {"item_id": "10", "item_name": "Counter-Strike", "playtime_forever": 10},
            ],
        },
    ]
    return _make_user_items_gz(records)


@pytest.fixture()
def steam_games_bytes() -> bytes:
    records = [
        {"id": "10", "title": "Counter-Strike", "genres": ["Action"], "tags": ["FPS"]},
        {"id": "30", "title": "Dota 2",         "genres": ["Strategy"], "tags": ["MOBA"]},
    ]
    return _make_steam_games_gz(records)


def _build_service(
    raw_dir: Path,
    processed_dir: Path,
    user_bytes: bytes,
    steam_bytes: bytes,
):
    """
    Write synthetic gzip files to disk and instantiate CleanDataService
    pointing at them. Returns the service instance.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    user_path = raw_dir / "australian_users_items.json.gz"
    steam_path = raw_dir / "steam_games.json.gz"
    user_path.write_bytes(user_bytes)
    steam_path.write_bytes(steam_bytes)

    cfg = {
        "data_ingestion": {
            "user_item_dataset_download_url": (
                f"https://example.com/{user_path.name}"
            ),
            "steam_game_dataset_download_url": (
                f"https://example.com/{steam_path.name}"
            ),
        },
        "data_cleaning": {
            "raw_data_dir": str(raw_dir),
            "root_dir": str(processed_dir),
        },
    }

    # Patch load_config so __init__ doesn't read an actual YAML file
    with patch("training.data_cleaning.load_config", return_value=cfg):
        from training.data_cleaning import CleanDataService
        return CleanDataService(config_path="dummy.yaml")


# ---------------------------------------------------------------------------
# Tests: run() happy path
# ---------------------------------------------------------------------------

class TestCleanDataServiceRun:

    def test_output_file_created(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        output_path = svc.run()
        assert os.path.exists(output_path)

    def test_output_is_valid_csv(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        output_path = svc.run()
        df = pd.read_csv(output_path)
        assert not df.empty

    def test_all_user_interactions_present(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        """Left-join: all 3 user-item rows must appear in the output."""
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        df = pd.read_csv(svc.run())
        # user_001 has 2 items, user_002 has 1
        assert len(df) == 3

    def test_user_ids_preserved(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        df = pd.read_csv(svc.run())
        assert set(df["user_id"].unique()) == {"user_001", "user_002"}

    def test_metadata_joined_for_known_items(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        """Counter-Strike (id=10) exists in steam games -> title column should be populated."""
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        df = pd.read_csv(svc.run())
        cs_rows = df[df["item_name"] == "Counter-Strike"]
        assert not cs_rows.empty
        assert cs_rows["title"].notna().all()

    def test_unmatched_user_items_kept_with_nan_metadata(
        self, raw_dir, processed_dir, steam_games_bytes
    ):
        """Portal (id=20) is NOT in steam_games -> row kept but metadata is NaN."""
        user_bytes = _make_user_items_gz([
            {
                "user_id": "user_001",
                "items": [
                    {"item_id": "20", "item_name": "Portal", "playtime_forever": 50},
                ],
            }
        ])
        svc = _build_service(raw_dir, processed_dir, user_bytes, steam_games_bytes)
        df = pd.read_csv(svc.run())
        portal_row = df[df["item_name"] == "Portal"]
        assert len(portal_row) == 1
        assert pd.isna(portal_row.iloc[0]["title"])

    def test_required_columns_present(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        df = pd.read_csv(svc.run())
        for col in ("user_id", "item_id", "item_name", "playtime"):
            assert col in df.columns, f"Missing column: {col}"

    def test_item_id_column_is_string_type(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        """item_id must be str to ensure the merge key matches across both datasets."""
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        df = pd.read_csv(svc.run())
        assert df["item_id"].dtype == object  # pandas string columns are 'object'

    def test_returns_path_to_output_file(self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes):
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)
        result = svc.run()
        assert result.endswith(".csv")
        assert os.path.exists(result)


# ---------------------------------------------------------------------------
# Tests: idempotency (skip if output already exists)
# ---------------------------------------------------------------------------

class TestSkipIfExists:

    def test_skips_processing_when_output_exists(
        self, raw_dir, processed_dir, user_items_bytes, steam_games_bytes
    ):
        svc = _build_service(raw_dir, processed_dir, user_items_bytes, steam_games_bytes)

        # Create the output file in advance (simulates a previous run)
        expected_output = os.path.join(
            str(processed_dir), "australian_users_items_merged.csv"
        )
        Path(expected_output).write_text("already,here\n1,2\n")

        result = svc.run()

        assert result == expected_output
        # The existing content must not be overwritten
        assert Path(result).read_text().startswith("already,here")


# ---------------------------------------------------------------------------
# Tests: malformed input lines are skipped, not fatal
# ---------------------------------------------------------------------------

class TestMalformedInputHandling:

    def test_malformed_user_item_line_is_skipped(
        self, raw_dir, processed_dir, steam_games_bytes
    ):
        """A corrupt line in the user-items file must be skipped gracefully."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            # one valid line
            valid = repr({
                "user_id": "user_001",
                "items": [{"item_id": "10", "item_name": "CS", "playtime_forever": 5}],
            }) + "\n"
            gz.write(valid.encode())
            # one corrupt line
            gz.write(b"THIS IS NOT VALID PYTHON LITERAL\n")

        svc = _build_service(raw_dir, processed_dir, buf.getvalue(), steam_games_bytes)
        df = pd.read_csv(svc.run())
        assert len(df) == 1  # only the valid row

    def test_malformed_steam_game_line_is_skipped(
        self, raw_dir, processed_dir, user_items_bytes
    ):
        """A corrupt line in the steam-games file must be skipped."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            valid = repr({"id": "10", "title": "CS", "genres": ["Action"], "tags": []}) + "\n"
            gz.write(valid.encode())
            gz.write(b"CORRUPT_LINE\n")

        svc = _build_service(raw_dir, processed_dir, user_items_bytes, buf.getvalue())
        # Should complete without raising
        df = pd.read_csv(svc.run())
        assert not df.empty


# ---------------------------------------------------------------------------
# Tests: missing raw files raise an appropriate error
# ---------------------------------------------------------------------------

class TestMissingFiles:

    def test_missing_user_items_file_raises(self, raw_dir, processed_dir, steam_games_bytes):
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Only write the steam games file, not user items
        steam_path = raw_dir / "steam_games.json.gz"
        steam_path.write_bytes(steam_games_bytes)

        cfg = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/australian_users_items.json.gz",
                "steam_game_dataset_download_url": f"https://example.com/steam_games.json.gz",
            },
            "data_cleaning": {
                "raw_data_dir": str(raw_dir),
                "root_dir": str(processed_dir),
            },
        }

        with patch("training.data_cleaning.load_config", return_value=cfg):
            from training.data_cleaning import CleanDataService
            with pytest.raises(CustomException):
                CleanDataService(config_path="dummy.yaml")

    def test_missing_steam_games_file_raises(self, raw_dir, processed_dir, user_items_bytes):
        raw_dir.mkdir(parents=True, exist_ok=True)
        user_path = raw_dir / "australian_users_items.json.gz"
        user_path.write_bytes(user_items_bytes)

        cfg = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/australian_users_items.json.gz",
                "steam_game_dataset_download_url": "https://example.com/steam_games.json.gz",
            },
            "data_cleaning": {
                "raw_data_dir": str(raw_dir),
                "root_dir": str(processed_dir),
            },
        }

        with patch("training.data_cleaning.load_config", return_value=cfg):
            from training.data_cleaning import CleanDataService
            with pytest.raises(CustomException):
                CleanDataService(config_path="dummy.yaml")
