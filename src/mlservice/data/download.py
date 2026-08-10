"""Fetch the Diabetes 130-US Hospitals dataset and verify it byte-for-byte.

Reproducibility here is deliberate and cheap: rather than committing ~20 MB of
patient-level records so everyone works from the same bytes, we commit a SHA256
and verify the download against it. Same guarantee, none of the cost, and no
clinical records in git history.

The checksum is recorded on first download and enforced on every subsequent
one. If UCI ever republishes the archive, verification fails loudly rather than
silently changing every number downstream — which is exactly what you want,
because "the dataset changed under me" is otherwise an extremely hard bug to
see.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from mlservice.config import PROJECT_ROOT, get_settings
from mlservice.logging_ import get_logger

log = get_logger(__name__)

CHECKSUM_FILE: Path = PROJECT_ROOT / "data" / "checksums.txt"

#: Files expected inside the archive. Named explicitly so a changed archive
#: layout is caught here rather than as a confusing FileNotFoundError later.
EXPECTED_MEMBERS = ("diabetic_data.csv", "IDS_mapping.csv")

_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class DownloadResult:
    archive: Path
    sha256: str
    extracted: tuple[Path, ...]
    was_cached: bool


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def read_recorded_checksum(name: str) -> str | None:
    """Return the checksum recorded for ``name``, if the file records one."""
    if not CHECKSUM_FILE.is_file():
        return None
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) == 2 and parts[1] == name:
            return parts[0]
    return None


def record_checksum(name: str, digest: str) -> None:
    """Record a checksum, creating the file with an explanatory header."""
    header = (
        "# SHA256 checksums for the raw dataset archive.\n"
        "#\n"
        "# Committed so the pipeline is reproducible without committing\n"
        "# patient-level data. Verified on every download; a mismatch means the\n"
        "# upstream archive changed and every downstream number would silently\n"
        "# change with it.\n"
        "#\n"
        "# Format: <sha256>  <filename>\n"
    )
    existing = ""
    if CHECKSUM_FILE.is_file():
        existing = "".join(
            line + "\n"
            for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#") and not line.strip().endswith(name)
        )
    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_FILE.write_text(f"{header}{existing}{digest}  {name}\n", encoding="utf-8")


class ChecksumMismatchError(RuntimeError):
    """The downloaded archive does not match the recorded checksum."""


def download(*, force: bool = False) -> DownloadResult:
    """Download, verify and extract the dataset.

    Skips the network entirely when a verified copy is already present, so
    ``make audit`` is fast and works offline once seeded.
    """
    settings = get_settings()
    raw_dir = settings.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = settings.data.source_url
    if not url:
        raise ValueError("data.source_url is empty — cannot download the dataset.")

    archive = raw_dir / "diabetes-130-us-hospitals.zip"
    recorded = read_recorded_checksum(archive.name)

    if archive.is_file() and not force:
        digest = sha256_of(archive)
        if recorded is None:
            record_checksum(archive.name, digest)
            log.info("checksum_recorded", file=archive.name, sha256=digest)
        elif digest != recorded:
            raise ChecksumMismatchError(
                f"{archive} does not match the recorded checksum.\n"
                f"  expected {recorded}\n  actual   {digest}\n"
                "Delete the file and re-download, or investigate why it changed."
            )
        log.info("download_skipped_cached", file=archive.name, sha256=digest)
        return DownloadResult(archive, digest, _extract(archive, raw_dir), was_cached=True)

    log.info("download_started", url=url)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    with archive.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=_CHUNK):
            fh.write(chunk)

    digest = sha256_of(archive)

    if recorded is None:
        record_checksum(archive.name, digest)
        log.info(
            "checksum_recorded",
            file=archive.name,
            sha256=digest,
            note="first download — commit data/checksums.txt to pin it",
        )
    elif digest != recorded:
        raise ChecksumMismatchError(
            f"Downloaded archive does not match the committed checksum.\n"
            f"  expected {recorded}\n  actual   {digest}\n"
            "The upstream dataset changed. Do NOT silently accept this — every "
            "metric in docs/DATA_AUDIT.md was computed against the recorded "
            "version."
        )

    size_mb = archive.stat().st_size / 1e6
    log.info("download_complete", file=archive.name, size_mb=round(size_mb, 2), sha256=digest)

    return DownloadResult(archive, digest, _extract(archive, raw_dir), was_cached=False)


def _extract(archive: Path, dest: Path) -> tuple[Path, ...]:
    """Extract expected members, refusing anything that escapes ``dest``."""
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for member in EXPECTED_MEMBERS:
            match = next((n for n in names if n.endswith(member)), None)
            if match is None:
                raise FileNotFoundError(
                    f"{member} not found in {archive.name}. Archive contents: {names}"
                )
            # Zip-slip guard: a crafted archive can otherwise write outside dest.
            target = (dest / Path(match).name).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f"Refusing to extract outside {dest}: {match}")
            with zf.open(match) as src, target.open("wb") as out:
                out.write(src.read())
            extracted.append(target)
            log.info("extracted", file=target.name, size_kb=round(target.stat().st_size / 1e3))
    return tuple(extracted)


__all__ = [
    "CHECKSUM_FILE",
    "ChecksumMismatchError",
    "DownloadResult",
    "download",
    "read_recorded_checksum",
    "record_checksum",
    "sha256_of",
]
