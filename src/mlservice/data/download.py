"""Fetch the Lending Club accepted-loans file and verify it byte-for-byte.

Reproducibility here is deliberate and cheap: rather than committing 1.6 GB of
loan records so everyone works from the same bytes, we commit a SHA256 and
verify the download against it. Same guarantee, none of the cost, and no
borrower-level data in git history.

**The checksum matters more here than it did for the medical dataset.** Lending
Club withdrew the official download, so the source is a community mirror on the
Hugging Face hub. A mirror can change, disappear, or be substituted, and any of
those would silently alter every number downstream. Pinning the bytes converts
that from an invisible problem into a loud one:

    ChecksumMismatchError: ... does not match the recorded checksum

Treat the checksum as the source of truth and the URL as merely where the bytes
happened to live.

The file is a plain CSV, not an archive, so there is no extraction step — the
medical version unpacked a zip. Downloads stream to a temporary file and are
renamed into place only after verification, so an interrupted download can
never masquerade as a cached one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests

from mlservice.config import PROJECT_ROOT, get_settings
from mlservice.logging_ import get_logger

log = get_logger(__name__)

CHECKSUM_FILE: Path = PROJECT_ROOT / "data" / "checksums.txt"

_CHUNK = 1 << 22  # 4 MiB — the file is ~1.6 GB, so larger chunks are worth it

#: Columns that must be present. Checked immediately after download so a
#: substituted or truncated mirror fails here, with a clear message, rather
#: than as a confusing KeyError somewhere in cleaning.
REQUIRED_COLUMNS = ("issue_d", "loan_status", "term", "loan_amnt", "grade", "addr_state")


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    size_bytes: int
    was_cached: bool

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / 1_048_576, 1)


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
        "# SHA256 checksums for the raw dataset.\n"
        "#\n"
        "# Committed so the pipeline is reproducible without committing 1.6 GB\n"
        "# of loan records. Verified on every download.\n"
        "#\n"
        "# This matters more than usual: Lending Club withdrew the official\n"
        "# download, so the source is a community mirror. A mirror that changes\n"
        "# under us would alter every number downstream in silence. Pinning the\n"
        "# bytes makes that failure loud instead.\n"
        "#\n"
        "# Format: <sha256>  <filename>\n"
    )
    existing = "".join(
        line + "\n"
        for line in (
            CHECKSUM_FILE.read_text(encoding="utf-8").splitlines()
            if CHECKSUM_FILE.is_file()
            else []
        )
        if line.strip() and not line.startswith("#") and not line.strip().endswith(name)
    )
    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_FILE.write_text(f"{header}{existing}{digest}  {name}\n", encoding="utf-8")


class ChecksumMismatchError(RuntimeError):
    """The downloaded file does not match the recorded checksum."""


class SchemaMismatchError(RuntimeError):
    """The downloaded file is missing columns the pipeline depends on."""


def verify_schema(path: Path) -> tuple[str, ...]:
    """Read only the header row and confirm the columns we rely on are there.

    Reading the header alone keeps this cheap on a 1.6 GB file, and catching a
    substituted mirror *here* means the error names the real problem instead of
    surfacing as a KeyError three modules later.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip()

    columns = tuple(c.strip().strip('"') for c in header.split(","))
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise SchemaMismatchError(
            f"{path.name} is missing required columns: {missing}. "
            f"Found {len(columns)} columns beginning {columns[:5]}. "
            "The mirror may have changed or the download may be truncated."
        )
    return columns


def download(*, force: bool = False) -> DownloadResult:
    """Download, verify and return the raw dataset.

    Skips the network entirely when a verified copy is already present, so
    ``make audit`` is fast and works offline once seeded.
    """
    settings = get_settings()
    raw_dir = settings.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = settings.data.source_url
    if not url:
        raise ValueError("data.source_url is empty — cannot download the dataset.")

    target = raw_dir / settings.data.archive_name
    recorded = read_recorded_checksum(target.name)

    if target.is_file() and not force:
        digest = sha256_of(target)
        if recorded is None:
            record_checksum(target.name, digest)
            log.info("checksum_recorded", file=target.name, sha256=digest)
        elif digest != recorded:
            raise ChecksumMismatchError(
                f"{target} does not match the recorded checksum.\n"
                f"  expected {recorded}\n  actual   {digest}\n"
                "The upstream mirror may have changed. Investigate before "
                "re-recording — every downstream number depends on these bytes."
            )
        verify_schema(target)
        size = target.stat().st_size
        log.info("download_skipped_cached", file=target.name, sha256=digest, size_mb=size / 1e6)
        return DownloadResult(target, digest, size, was_cached=True)

    log.info("download_started", url=url, expected_mb=1598)

    # Streamed to a partial file and renamed only after verification. An
    # interrupted download must never be mistaken for a cached one on the next
    # run — that failure is silent and extremely annoying to diagnose.
    partial = target.with_suffix(target.suffix + ".partial")
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()

    written = 0
    with partial.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=_CHUNK):
            if not chunk:
                continue
            fh.write(chunk)
            written += len(chunk)
            if written % (256 * 1024 * 1024) < _CHUNK:
                log.info("download_progress", mb=round(written / 1e6))

    digest = sha256_of(partial)
    if recorded is not None and digest != recorded:
        partial.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"downloaded file does not match the recorded checksum.\n"
            f"  expected {recorded}\n  actual   {digest}\n"
            "The mirror changed. The partial download has been deleted."
        )

    partial.replace(target)
    verify_schema(target)

    if recorded is None:
        record_checksum(target.name, digest)
        log.info("checksum_recorded", file=target.name, sha256=digest)

    size = target.stat().st_size
    log.info("download_complete", file=target.name, sha256=digest, size_mb=size / 1e6)
    return DownloadResult(target, digest, size, was_cached=False)


__all__ = [
    "REQUIRED_COLUMNS",
    "ChecksumMismatchError",
    "DownloadResult",
    "SchemaMismatchError",
    "download",
    "read_recorded_checksum",
    "record_checksum",
    "sha256_of",
    "verify_schema",
]
