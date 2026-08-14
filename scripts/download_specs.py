"""
Downloads a curated set of 3GPP Release-17 specifications from the official
3GPP FTP server and extracts the .docx source files.

Why a fixed URL list instead of a crawler: the 3GPP FTP directory listing
changes over time (new corrections are added to the SAME release for years
after freeze), so we pin exact filenames here. Check the directory listing
manually (e.g. https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/) if
you need to refresh these to a newer maintenance version later.

Release-letter note: the FTP filename encodes the release as a letter
(...f=14, g=15, h=16, i=17...) -- but ALWAYS verify against the document's
own cover-page version string ("V17.x.x") after downloading, since it's easy
to get the letter-to-release mapping wrong by guessing from dates alone.

Usage:
    python scripts/download_specs.py
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_ZIP_DIR = ROOT / "data" / "raw" / "zips"
RAW_DOCX_DIR = ROOT / "data" / "raw" / "docx"

BASE_URL = "https://www.3gpp.org/ftp/Specs/archive"

# 3GPP FTP blocks requests without a browser-like User-Agent (Cloudflare 403).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class SpecFile:
    spec_id: str        # e.g. "23.501"
    series: str          # e.g. "23_series"
    filename: str         # e.g. "23501-hf0.zip"
    title: str


# Curated 5G SA "core" set, pinned to the latest Release 17 maintenance drop
# available as of 2026-08-14. Verified against each docx's own cover-page
# version string ("3GPP TS x V17.y.z"), not just the FTP filename letter.
SPECS: list[SpecFile] = [
    SpecFile("23.501", "23_series", "23501-hf0.zip", "System architecture for the 5G System (5GS)"),
    SpecFile("23.502", "23_series", "23502-hf0.zip", "Procedures for the 5G System (5GS)"),
    SpecFile("23.503", "23_series", "23503-hb0.zip", "Policy and charging control framework for the 5G System"),
    SpecFile("24.501", "24_series", "24501-hi1.zip", "NAS protocol for 5G System (5GS)"),
    SpecFile("24.301", "24_series", "24301-hc0.zip", "NAS protocol for Evolved Packet System (EPS)"),
    SpecFile("38.300", "38_series", "38300-hh0.zip", "NR and NG-RAN Overall Description"),
    SpecFile("38.331", "38_series", "38331-hh0.zip", "NR Radio Resource Control (RRC) protocol specification"),
    SpecFile("33.501", "33_series", "33501-hg0.zip", "Security architecture and procedures for 5G System"),
    SpecFile("29.500", "29_series", "29500-hg0.zip", "5G System; Technical Realization of Service Based Architecture"),
    SpecFile("29.518", "29_series", "29518-hi0.zip", "5G System; Access and Mobility Management Services"),
    SpecFile("38.401", "38_series", "38401-ha0.zip", "NG-RAN; Architecture description"),
    SpecFile("38.413", "38_series", "38413-hf0.zip", "NG-RAN; NG Application Protocol (NGAP)"),
]


def download_zip(spec: SpecFile) -> Path:
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_ZIP_DIR / spec.filename
    if dest.exists():
        print(f"[skip] {spec.spec_id}: already downloaded -> {dest.name}")
        return dest

    url = f"{BASE_URL}/{spec.series}/{spec.spec_id}/{spec.filename}"
    print(f"[get ] {spec.spec_id}: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"       saved {len(resp.content) / 1024:.0f} KB -> {dest.name}")
    return dest


def extract_docx(spec: SpecFile, zip_path: Path) -> list[Path]:
    RAW_DOCX_DIR.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".docx"):
                continue
            out_path = RAW_DOCX_DIR / f"{spec.spec_id}.docx"
            with zf.open(name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            extracted.append(out_path)
    if not extracted:
        print(f"[warn] {spec.spec_id}: no .docx found inside {zip_path.name}")
    return extracted


def main() -> None:
    print(f"Downloading {len(SPECS)} specs into {RAW_ZIP_DIR}\n")
    for spec in SPECS:
        zip_path = download_zip(spec)
        extract_docx(spec, zip_path)
    print("\nDone. Raw .docx files are in:", RAW_DOCX_DIR)


if __name__ == "__main__":
    main()
