"""
Unit tests for training.data_ingestion.LoadDataService

Tests cover:
- __init__: directory creation, URL list population from config
- run(): skips download when file already present
- run(): calls download_file for missing files
- run(): handles multiple URLs independently
- Error propagation via CustomException

No real HTTP calls or filesystem writes outside tmp_path are made.

Run with:
    pytest tests/training/test_data_ingestion.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from training.utils.exception import CustomException


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

def _make_config(raw_data_dir: str) -> dict:
    return {
        "data_ingestion": {
            "user_item_dataset_download_url": (
                "https://example.com/australian_users_items.json.gz"
            ),
            "steam_game_dataset_download_url": (
                "https://example.com/steam_games.json.gz"
            ),
            "raw_data_dir": raw_data_dir,
        }
    }


def _make_service(raw_data_dir: str):
    """Instantiate LoadDataService with a mocked config loader."""
    cfg = _make_config(raw_data_dir)
    with patch("training.data_ingestion.load_config", return_value=cfg):
        from training.data_ingestion import LoadDataService
        return LoadDataService(config_path="dummy.yaml")


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------

class TestLoadDataServiceInit:

    def test_raw_data_dir_created_on_init(self, tmp_path):
        raw_dir = str(tmp_path / "new_raw_dir")
        assert not os.path.exists(raw_dir)
        _make_service(raw_dir)
        assert os.path.isdir(raw_dir)

    def test_urls_populated_from_config(self, tmp_path):
        svc = _make_service(str(tmp_path))
        assert len(svc.urls) == 2
        assert any("australian_users_items" in u for u in svc.urls)
        assert any("steam_games" in u for u in svc.urls)

    def test_raises_if_raw_data_dir_missing_from_config(self, tmp_path):
        cfg = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/a.gz",
                "steam_game_dataset_download_url": "https://example.com/b.gz",
                # raw_data_dir intentionally absent
            }
        }
        with patch("training.data_ingestion.load_config", return_value=cfg):
            from training.data_ingestion import LoadDataService
            with pytest.raises(CustomException):
                LoadDataService(config_path="dummy.yaml")

    def test_default_urls_used_when_not_in_config(self, tmp_path):
        """If the config keys are absent, fall back to the hard-coded Steam URLs."""
        cfg = {
            "data_ingestion": {
                "raw_data_dir": str(tmp_path),
                # URL keys intentionally absent
            }
        }
        with patch("training.data_ingestion.load_config", return_value=cfg):
            from training.data_ingestion import LoadDataService
            svc = LoadDataService(config_path="dummy.yaml")
        assert len(svc.urls) == 2
        assert all(u.startswith("https://") for u in svc.urls)


# ---------------------------------------------------------------------------
# Tests: run() -- download logic
# ---------------------------------------------------------------------------

class TestLoadDataServiceRun:

    def test_skips_download_when_file_exists(self, tmp_path):
        svc = _make_service(str(tmp_path))
        # Pre-create both expected files on disk
        for url in svc.urls:
            filename = url.split("/")[-1]
            (tmp_path / filename).write_bytes(b"dummy")

        with patch("training.data_ingestion.download_file") as mock_dl:
            svc.run()
            mock_dl.assert_not_called()

    def test_downloads_missing_file(self, tmp_path):
        svc = _make_service(str(tmp_path))
        # Only pre-create the first file; second must be downloaded
        first_url, second_url = svc.urls
        first_file = tmp_path / first_url.split("/")[-1]
        first_file.write_bytes(b"dummy")

        with patch("training.data_ingestion.download_file") as mock_dl:
            svc.run()
            mock_dl.assert_called_once_with(second_url, str(tmp_path))

    def test_downloads_all_files_when_none_exist(self, tmp_path):
        svc = _make_service(str(tmp_path))
        with patch("training.data_ingestion.download_file") as mock_dl:
            svc.run()
            assert mock_dl.call_count == 2
            called_urls = {c.args[0] for c in mock_dl.call_args_list}
            assert called_urls == set(svc.urls)

    def test_download_file_receives_correct_directory(self, tmp_path):
        svc = _make_service(str(tmp_path))
        with patch("training.data_ingestion.download_file") as mock_dl:
            svc.run()
            for c in mock_dl.call_args_list:
                assert c.args[1] == str(tmp_path)

    def test_filename_derived_from_url(self, tmp_path):
        """
        Verifies the skip-check uses only the filename from the URL, not the
        full path, so a file named steam_games.json.gz in the right directory
        is correctly detected as present.
        """
        svc = _make_service(str(tmp_path))
        # Create a file matching the URL's basename
        for url in svc.urls:
            filename = url.split("/")[-1]
            (tmp_path / filename).write_bytes(b"content")

        with patch("training.data_ingestion.download_file") as mock_dl:
            svc.run()
            mock_dl.assert_not_called()

    def test_run_wraps_exceptions_as_custom_exception(self, tmp_path):
        """If download_file raises, run() must re-raise as CustomException."""
        svc = _make_service(str(tmp_path))
        with patch(
            "training.data_ingestion.download_file",
            side_effect=ConnectionError("network unavailable"),
        ):
            with pytest.raises(CustomException):
                svc.run()


# ---------------------------------------------------------------------------
# Tests: URL handling
# ---------------------------------------------------------------------------

class TestURLHandling:

    def test_filename_extracted_from_url_correctly(self, tmp_path):
        cfg = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://host.com/path/to/my_file.json.gz",
                "steam_game_dataset_download_url": "https://host.com/steam_games.json.gz",
                "raw_data_dir": str(tmp_path),
            }
        }
        with patch("training.data_ingestion.load_config", return_value=cfg):
            from training.data_ingestion import LoadDataService
            svc = LoadDataService(config_path="dummy.yaml")

        # Pre-create both files so no download is triggered
        (tmp_path / "my_file.json.gz").write_bytes(b"x")
        (tmp_path / "steam_games.json.gz").write_bytes(b"x")

        with patch("training.data_ingestion.download_file") as mock_dl:
            svc.run()
            mock_dl.assert_not_called()

    def test_each_url_processed_independently(self, tmp_path):
        svc = _make_service(str(tmp_path))
        download_order = []

        def fake_download(url, dest):
            download_order.append(url)

        with patch("training.data_ingestion.download_file", side_effect=fake_download):
            svc.run()

        assert download_order == svc.urls
