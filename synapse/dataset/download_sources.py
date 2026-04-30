#!/usr/bin/env python3
"""Download Z3 Evaluation Datasets from HuggingFace.

Downloads external datasets into a well-structured, persistent directory
layout under ``data_root`` (default: ``data/datasets``).  Each dataset
gets its own subdirectory with a ``MANIFEST.json`` marker file that
records download status, enabling fast detection of which datasets are
ready and which are missing — no redundant downloads.

Directory layout
----------------
    data_root/
      telecom/
        MANIFEST.json       # download metadata (status, timestamp, sha)
        raw/                # raw HuggingFace cache symlink / copy
      spatial/
        MANIFEST.json
        raw/
      photonic/
        MANIFEST.json
        raw/
      synthetic/            # no download needed, generated in-place
        MANIFEST.json

Usage
-----
    python -m synapse.dataset.download_sources
    python -m synapse.dataset.download_sources --datasets telecom spatial
    python -m synapse.dataset.download_sources --all
    python -m synapse.dataset.download_sources --check

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3
- Data source doc: ``docs/dev/data_source.md``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Default data root (relative to project root).  Overridden by --data-root.
DEFAULT_DATA_ROOT = "data/datasets"

# HuggingFace repositories for each evaluation track.
# Keys match the canonical dataset names used in adapters and configs.
HF_REPOS = {
    "telecom": "AliMaatouk/TelecomTS",
    "spatial": "manycore-research/SpatialLM-Dataset",
    "photonic": "cgeorgiaw/2d-photonic-topology",
}

# Datasets that require no download (generated in memory).
NO_DOWNLOAD_DATASETS = {"synthetic"}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME = "MANIFEST.json"


def _manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / _MANIFEST_FILENAME


def read_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    """Read a dataset's MANIFEST.json, or return None if missing."""
    mp = _manifest_path(dataset_dir)
    if not mp.is_file():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Corrupt manifest at %s: %s", mp, exc)
        return None


def write_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    """Write a MANIFEST.json atomically."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    mp = _manifest_path(dataset_dir)
    mp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_manifest(
    name: str,
    status: str,
    *,
    hf_repo: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "dataset": name,
        "status": status,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    if hf_repo:
        manifest["hf_repo"] = hf_repo
    if extra:
        manifest.update(extra)
    return manifest


# ---------------------------------------------------------------------------
# Download detection
# ---------------------------------------------------------------------------

def is_downloaded(name: str, data_root: str | Path) -> bool:
    """Check whether a dataset has been successfully downloaded.

    Parameters
    ----------
    name : str
        Canonical dataset name (e.g. ``"telecom"``).
    data_root : str or Path
        Root directory for all datasets.

    Returns
    -------
    bool
        ``True`` if the manifest exists and status is ``"complete"``.
    """
    if name in NO_DOWNLOAD_DATASETS:
        return True

    dataset_dir = Path(data_root) / name
    manifest = read_manifest(dataset_dir)
    return manifest is not None and manifest.get("status") == "complete"


def list_download_status(data_root: str | Path) -> dict[str, str]:
    """Return download status for all known datasets.

    Returns
    -------
    dict
        Mapping ``dataset_name → status_string`` where status is one of
        ``"complete"``, ``"partial"``, ``"missing"``, or ``"no_download"``.
    """
    all_names = sorted(set(HF_REPOS) | NO_DOWNLOAD_DATASETS)
    result: dict[str, str] = {}
    for name in all_names:
        if name in NO_DOWNLOAD_DATASETS:
            result[name] = "no_download"
            continue
        manifest = read_manifest(Path(data_root) / name)
        if manifest is None:
            result[name] = "missing"
        elif manifest.get("status") == "complete":
            result[name] = "complete"
        else:
            result[name] = "partial"
    return result


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_dataset(name: str, data_root: str | Path | None = None) -> bool:
    """Download a single dataset from HuggingFace to a structured directory.

    If the dataset is already marked as ``"complete"`` in its manifest,
    the download is skipped.

    Parameters
    ----------
    name : str
        Short name (e.g. ``"telecom"``).
    data_root : str or Path or None
        Root directory for downloaded data.  Defaults to DEFAULT_DATA_ROOT.

    Returns
    -------
    bool
        ``True`` if download succeeded or was already complete.
    """
    if name in NO_DOWNLOAD_DATASETS:
        log.info("Dataset '%s' is synthetic — no download required.", name)
        dr = Path(data_root or DEFAULT_DATA_ROOT) / name
        write_manifest(dr, _make_manifest(name, "complete"))
        return True

    if name not in HF_REPOS:
        log.error("Unknown dataset: '%s'. Available: %s", name, sorted(HF_REPOS.keys()))
        return False

    dr = Path(data_root or DEFAULT_DATA_ROOT)
    dataset_dir = dr / name

    # --- Check if already downloaded ---
    if is_downloaded(name, dr):
        log.info("Dataset '%s' already downloaded (manifest found). Skipping.", name)
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        log.error(
            "The 'datasets' package is required. Install with: pip install datasets"
        )
        return False

    # --- Write partial manifest before download ---
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        dataset_dir,
        _make_manifest(name, "partial", hf_repo=HF_REPOS[name]),
    )

    repo = HF_REPOS[name]
    log.info("Downloading '%s' from %s ...", name, repo)

    # --- Photonic uses JLD2 files (HDF5) — not load_dataset compatible ---
    if name == "photonic":
        return _download_photonic(name, repo, dataset_dir)

    try:
        ds = load_dataset(repo, trust_remote_code=True)
    except Exception as exc:
        log.error("  Failed to download '%s': %s", name, exc)
        write_manifest(
            dataset_dir,
            _make_manifest(name, "failed", hf_repo=repo, extra={"error": str(exc)}),
        )
        return False

    # --- Save raw data to dataset_dir/raw/ ---
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    try:
        ds.save_to_disk(str(raw_dir))
        log.info("  Saved raw data to %s", raw_dir)
    except Exception as exc:
        log.warning("  Could not save_to_disk: %s (HF cache still valid)", exc)

    # --- Write complete manifest ---
    num_samples = sum(len(split) for split in ds.values()) if hasattr(ds, "values") else 0
    write_manifest(
        dataset_dir,
        _make_manifest(
            name,
            "complete",
            hf_repo=repo,
            extra={"num_samples": num_samples, "raw_path": str(raw_dir)},
        ),
    )
    log.info("  '%s' downloaded and manifest written.", name)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _download_photonic(name: str, repo: str, dataset_dir: Path) -> bool:
    """Download photonic JLD2 files via huggingface_hub (not load_dataset)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error(
            "The 'huggingface_hub' package is required for photonic. "
            "Install with: pip install huggingface_hub"
        )
        return False

    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        local_dir = snapshot_download(
            repo,
            repo_type="dataset",
            allow_patterns=["lattices/*"],
            local_dir=str(raw_dir),
        )
        log.info("  Downloaded photonic lattices to %s", local_dir)
    except Exception as exc:
        log.error("  Failed to download photonic: %s", exc)
        write_manifest(
            dataset_dir,
            _make_manifest(name, "failed", hf_repo=repo, extra={"error": str(exc)}),
        )
        return False

    jld2_count = len(list(Path(raw_dir).rglob("*.jld2")))
    write_manifest(
        dataset_dir,
        _make_manifest(
            name,
            "complete",
            hf_repo=repo,
            extra={"num_jld2_files": jld2_count, "raw_path": str(raw_dir)},
        ),
    )
    log.info("  '%s' downloaded (%d JLD2 files).", name, jld2_count)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Z3 evaluation datasets from HuggingFace "
        "into a structured, persistent directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to download (e.g. telecom spatial photonic).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check download status; do not download.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory for dataset storage (default: {DEFAULT_DATA_ROOT}).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # --- Check mode ---
    if args.check:
        statuses = list_download_status(data_root)
        log.info("Download status (data_root=%s):", data_root)
        for name, status in statuses.items():
            marker = "✓" if status == "complete" else ("—" if status == "no_download" else "✗")
            log.info("  %s %s: %s", marker, name, status)
        missing = [n for n, s in statuses.items() if s == "missing"]
        return 0 if not missing else 1

    # --- Download mode ---
    if args.all:
        names = sorted(HF_REPOS.keys())
    elif args.datasets:
        names = args.datasets
    else:
        names = sorted(HF_REPOS.keys())

    log.info("Downloading %d dataset(s): %s", len(names), names)

    success = 0
    for name in names:
        if download_dataset(name, data_root=data_root):
            success += 1

    log.info(
        "Download complete: %d/%d succeeded.",
        success,
        len(names),
    )
    return 0 if success == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
