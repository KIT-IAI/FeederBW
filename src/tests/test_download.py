import zipfile
from unittest.mock import MagicMock, patch

import pytest

from src.download.download import (
    FILE_MAP,
    ZENODO_RECORD_ID,
    _resolve_record_id,
    _unzip_and_cleanup,
    download_data,
)


def test_file_map_contains_all_entries():
    assert len(FILE_MAP) == 7
    assert "feeder_metadata" in FILE_MAP
    assert "weather_data" in FILE_MAP
    assert "001-040" in FILE_MAP
    assert "041-080" in FILE_MAP
    assert "081-120" in FILE_MAP
    assert "121-160" in FILE_MAP
    assert "161-200" in FILE_MAP


@patch("src.download.download._unzip_and_cleanup")
@patch("src.download.download.zget.download")
def test_invalid_file_raises_value_error(mock_zenodo, mock_unzip, tmp_path):
    with pytest.raises(ValueError, match="Invalid file identifiers"):
        download_data(files=["nonexistent"], output_dir=tmp_path)
    mock_zenodo.assert_not_called()


def test_resolve_record_id_none_returns_default():
    assert _resolve_record_id(None) == ZENODO_RECORD_ID


@patch("src.download.download.requests.get")
def test_resolve_record_id_valid_version(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hits": {
            "hits": [
                {"id": 17831177, "metadata": {"version": "1.0.0"}},
            ]
        }
    }
    mock_get.return_value = mock_resp

    result = _resolve_record_id("1.0.0")
    assert result == "17831177"


@patch("src.download.download.requests.get")
def test_resolve_record_id_invalid_version(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hits": {
            "hits": [
                {"id": 17831177, "metadata": {"version": "1.0.0"}},
            ]
        }
    }
    mock_get.return_value = mock_resp

    with pytest.raises(ValueError, match="Version '9.9.9' not found"):
        _resolve_record_id("9.9.9")


def test_unzip_and_cleanup(tmp_path):
    zip_path = tmp_path / "test_archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test_file.txt", "hello world")

    _unzip_and_cleanup(tmp_path)

    assert (tmp_path / "test_file.txt").read_text() == "hello world"
    assert not zip_path.exists()


@patch("src.download.download._unzip_and_cleanup")
@patch("src.download.download.zget.download")
def test_download_all_calls_zenodo_get(mock_zenodo, mock_unzip, tmp_path):
    download_data(files=None, output_dir=tmp_path)

    mock_zenodo.assert_called_once()
    kwargs = mock_zenodo.call_args.kwargs
    assert kwargs["record_or_doi"] == ZENODO_RECORD_ID
    assert kwargs["output_dir"] == tmp_path
    assert "file_glob" not in kwargs


@patch("src.download.download._unzip_and_cleanup")
@patch("src.download.download.zget.download")
def test_download_selective_calls_zenodo_get(mock_zenodo, mock_unzip, tmp_path):
    download_data(files=["feeder_metadata", "001-040"], output_dir=tmp_path)

    assert mock_zenodo.call_count == 2
    call_kwargs = [c.kwargs for c in mock_zenodo.call_args_list]

    assert any(kw.get("file_glob") == "feeder_metadata.csv" for kw in call_kwargs)
    assert any(kw.get("file_glob") == "feeder_measurement_data_001-040.zip" for kw in call_kwargs)
