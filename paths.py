import os
from pathlib import Path


def data_path(name: str) -> Path:
    return Path(os.getenv("DATA_DIR", ".")) / name
