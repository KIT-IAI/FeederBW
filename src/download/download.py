from pathlib import Path
import zipfile

import requests
from zenodo_get import zget

from src.config import DATA_DIR

ZENODO_RECORD_ID = "19550073"  # NOTE(m): this is version v1.1 
ZENODO_CONCEPT_ID = "17831176"

VALID_MEASUREMENT_RANGES = ("001-040", "041-080", "081-120", "121-160", "161-200")

FILE_MAP = {
    "feeder_metadata": "feeder_metadata.csv",
    "weather_data": "weather_data.parquet",
    **{r: f"feeder_measurement_data_{r}.zip" for r in VALID_MEASUREMENT_RANGES},
}


def download_data(
    files: list[str] | None = None,
    version: str | None = None,
    output_dir: str | Path = DATA_DIR,
    overwrite: bool = False,
) -> None:
    """Download FeederBW dataset files from Zenodo.

    Args:
        files: List of file identifiers to download. None downloads all files.
            Valid identifiers: "feeder_metadata", "weather_data",
            "001-040", "041-080", "081-120", "121-160", "161-200".
        version: Dataset version string (e.g. "1.0.0"). None uses the default
            record ID for the latest known version.
        output_dir: Directory to store downloaded files. Created if it doesn't exist.
        overwrite: If True, removes existing data before downloading. If False and
            the directory is not empty, stops with a message.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(f for f in output_dir.iterdir() if f.name != ".gitkeep"):
        if not overwrite:
            print(
                f"'{output_dir}' is not empty. Set overwrite=True to remove existing "
                "data and download again."
            )
            return
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    record_id = _resolve_record_id(version)

    if files is not None:
        invalid = [f for f in files if f not in FILE_MAP]
        if invalid:
            raise ValueError(
                f"Invalid file identifiers: {invalid}. "
                f"Valid options: {list(FILE_MAP.keys())}"
            )

    if files is None:
        zget.download(record_or_doi=record_id, output_dir=output_dir)
    else:
        for file_id in files:
            filename = FILE_MAP[file_id]
            zget.download(
                record_or_doi=record_id,
                output_dir=output_dir,
                file_glob=filename,
            )

    _unzip_and_cleanup(output_dir)


def _resolve_record_id(version: str | None) -> str:
    if version is None:
        return ZENODO_RECORD_ID

    resp = requests.get(
        f"https://zenodo.org/api/records/{ZENODO_CONCEPT_ID}/versions",
        params={"size": 100},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json()["hits"]["hits"]
    for hit in hits:
        if hit["metadata"]["version"] == version:
            return str(hit["id"])

    available = [h["metadata"]["version"] for h in hits]
    raise ValueError(
        f"Version '{version}' not found. Available versions: {available}"
    )


def _unzip_and_cleanup(output_dir: Path) -> None:
    for zip_path in output_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            # NOTE (m): Validate that no zip entry would extract outside the target directory to prevent path traversal vulnerabilities.
            for member in zf.namelist():
                member_path = (output_dir / member).resolve()
                if not member_path.is_relative_to(output_dir.resolve()):
                    raise ValueError(f"Zip entry '{member}' would extract outside target directory.")
            zf.extractall(output_dir)
        zip_path.unlink()