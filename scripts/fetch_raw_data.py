#!/usr/bin/env python3
"""Download untouched source files for the four Phase 0B datasets.

Idempotent: skips a file if it already exists under data/raw/. Never
modifies data/raw/ contents after download (append-only per DATA_CONTRACT.md).
Run manually; not part of the automated test suite (network access required).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES: dict[str, list[tuple[str, str]]] = {
    "bvel_percolation": [
        ("BVEL13k.zip", "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/BVEL13k.zip"),
    ],
    "bvse_hops": [
        (
            "nebBVSE122k.zip",
            "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/nebBVSE122k.zip",
        ),
    ],
    "dft_neb": [
        ("nebDFT2k.zip", "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/nebDFT2k.zip"),
    ],
    "obelix": [
        (
            "all.csv",
            "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all.csv",
        ),
        (
            "train.csv",
            "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.csv",
        ),
        (
            "test.csv",
            "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test.csv",
        ),
        (
            "all_cifs.zip",
            "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all_cifs.zip",
        ),
        (
            "data_README.md",
            "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/README.md",
        ),
    ],
}


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "sse-transfer-phase0b/1.0"})
    with urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        while chunk := resp.read(1024 * 1024):
            f.write(chunk)


def fetch_all(extract: bool = True) -> None:
    for dataset, files in SOURCES.items():
        out_dir = RAW / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename, url in files:
            dest = out_dir / filename
            if dest.exists():
                print(f"[skip] {dest} already exists")
            else:
                print(f"[get ] {url} -> {dest}")
                _download(url, dest)
            if extract and dest.suffix == ".zip":
                extract_dir = out_dir / "extracted"
                if extract_dir.exists():
                    print(f"[skip] {extract_dir} already extracted")
                    continue
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(extract_dir)
                print(f"[unzip] {dest} -> {extract_dir}")


if __name__ == "__main__":
    fetch_all(extract="--no-extract" not in sys.argv)
