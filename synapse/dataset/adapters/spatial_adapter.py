"""Spatial Geometry Adapter.

Loads the SpatialLM-Dataset from HuggingFace and produces a
``DatasetBundle`` for the Z3 geometric evaluation track.

The dataset contains 12,328 indoor scenes as 3D point clouds.
Z3 uses the ``DifferentiableHodgeProxy`` (L1) to identify structural
"holes" (doorways, windows, furniture voids) that distinguish room types.

Source: https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3, Track 2
- Data source doc: ``docs/dev/data_source.md`` §3
"""

from __future__ import annotations

import csv
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..preprocess.geometric import GeometricPreprocessor
from ..registry import register_adapter
from .base import DatasetBundle, DatasetSpec, Z3Adapter
from .persistence import get_prepared_cache_dir, load_prepared_bundle, save_prepared_bundle
from .split_utils import apply_split, extract_predefined_splits, try_load_from_local

log = logging.getLogger(__name__)


class SpatialAdapter(Z3Adapter):
    """Adapter for the SpatialLM 3D point-cloud dataset.

    Parameters
    ----------
    target_length : int
        Fixed number of points per sample after sub-sampling.
    train_ratio : float
        Fraction of data used for training (default 0.8).
    val_ratio : float
        Fraction of data used for validation (default 0.1).
    seed : int
        Random seed for reproducible splitting.
    max_samples : int or None
        Cap on total samples loaded (for smoke tests).
    data_root : str or None
        Root directory for downloaded datasets.  If set, the adapter
        will try loading from local disk before HuggingFace.
    """

    def __init__(
        self,
        *,
        target_length: int = 512,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: int | None = None,
        data_root: str | None = None,
    ) -> None:
        self._target_length = target_length
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        self._max_samples = max_samples
        self._data_root = data_root
        self._raw_repo_root: Path | None = None

        self._spec = DatasetSpec(
            name="spatial",
            modality="geometric",
            source="huggingface",
            hf_repo="manycore-research/SpatialLM-Dataset",
            input_dim=3,
            sequence_length=target_length,
            num_classes=15,
            task="classification",
            max_samples=max_samples,
            data_root=data_root,
        )

    # ---- Z3Adapter interface -----------------------------------------------

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    @property
    def input_dim(self) -> int:
        return self._spec.input_dim

    @property
    def num_classes(self) -> int:
        return self._spec.num_classes

    def load_splits(self) -> DatasetBundle:
        """Load SpatialLM and return splits (with disk caching).
        
        This is the primary entry point. It checks for prepared split
        files on disk before falling back to raw extraction.

        Returns
        -------
        DatasetBundle
        """
        # 1. Check for final prepared splits first
        cache_dir = Path(self._data_root or "data/datasets")
        cache_path = get_prepared_cache_dir(
            cache_dir, "spatial", self._spec, self._seed, self._train_ratio, self._val_ratio
        )
        
        bundle = load_prepared_bundle(cache_path, self._spec)
        if bundle is not None:
            return bundle

        # 2. If no prepared cache, load raw data (local or HF)
        log.info("No prepared cache found for SpatialLM. Starting extraction...")
        ds = self._load_data()
        split_result = extract_predefined_splits(
            ds,
            extract_array=self._extract_cloud,
            extract_label=self._extract_label,
            max_samples=self._max_samples,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
        )
        if split_result is not None:
            clouds, labels = split_result
        else:
            raw_clouds, raw_labels = self._extract_from_dataset(ds)
            clouds, labels = apply_split(
                raw_clouds,
                raw_labels,
                train_ratio=self._train_ratio,
                val_ratio=self._val_ratio,
                seed=self._seed,
            )

        # 4. Wrap and Preprocess
        preprocessor = GeometricPreprocessor(target_length=self._target_length)
        bundle = DatasetBundle(
            train_sequences=clouds["train"],
            train_labels=labels["train"],
            val_sequences=clouds["val"],
            val_labels=labels["val"],
            test_sequences=clouds["test"],
            test_labels=labels["test"],
            spec=self._resolved_spec_from_sequences(clouds["train"]),
        )
        bundle = preprocessor(bundle)

        # 5. Save the FINAL PREPARED data to disk for next time
        save_prepared_bundle(cache_path, bundle)
        
        return bundle

    # ------------------------------------------------------------------ #

    def _load_data(self):
        """Load the raw SpatialLM dataset object."""
        self._raw_repo_root = None

        local_repo_root = self._find_local_repo_root()
        if local_repo_root is not None:
            self._raw_repo_root = local_repo_root
            log.info("SpatialAdapter: loading metadata from local repo at %s", local_repo_root)
            return self._load_local_metadata(local_repo_root)

        # --- Try local disk first ---
        local_ds = try_load_from_local(self._data_root, "spatial")
        if local_ds is not None:
            log.info("SpatialAdapter: loading from local data_root=%s", self._data_root)
            return local_ds

        # --- Fall back to HuggingFace ---
        return self._load_from_huggingface()

    # Approximate number of PLY files per ZIP chunk.
    _PLY_PER_CHUNK: int = 2000

    def _load_from_huggingface(self):
        """Download SpatialLM from HuggingFace and extract PCD archives.

        SpatialLM stores point clouds in ZIP chunks (pcd/chunk_NNN.zip),
        not as individual files.  To avoid downloading the full ~38 GB,
        we first fetch ``split.csv`` (tiny), determine how many chunks
        are needed to cover *max_samples*, and only download those.

        Note: ``snapshot_download`` with ``allow_patterns`` does not
        respect the filter on XetHub-backed repos, so we use individual
        ``hf_hub_download`` calls instead.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_hub' package is required to load SpatialLM. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        repo_id = self._spec.hf_repo

        # Phase 1: download split.csv only (a few hundred KB)
        log.info("Downloading SpatialLM metadata from HuggingFace: %s", repo_id)
        split_csv_path = Path(hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="split.csv",
        ))
        repo_root = split_csv_path.parent

        # Phase 2: determine how many ZIP chunks we actually need
        needed_chunks = self._compute_needed_chunks(split_csv_path)
        chunk_files = [f"pcd/chunk_{i:03d}.zip" for i in range(needed_chunks)]

        log.info(
            "Downloading %d PCD ZIP chunks (~%.0f GB) from %s",
            needed_chunks,
            needed_chunks * 0.35,  # ~350 MB per chunk
            repo_id,
        )

        # Phase 3: download each chunk individually (allow_patterns doesn't work on XetHub)
        for chunk_file in chunk_files:
            log.info("  Downloading %s ...", chunk_file)
            hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=chunk_file,
            )

        # Extract ZIP archives if PLY files aren't already on disk
        self._ensure_pcd_extracted(repo_root)

        # Set repo root so _resolve_ply_path can find PLY files
        self._raw_repo_root = repo_root

        # Load metadata from split.csv
        return self._load_local_metadata(repo_root)

    def _compute_needed_chunks(self, split_csv_path: Path) -> int:
        """Return the number of ZIP chunks required for *max_samples*.

        If ``max_samples`` is ``None`` we still cap at a reasonable
        default (20 chunks ≈ 40K samples ≈ 7 GB) rather than pulling
        the entire 109-chunk / 38 GB dataset.
        """
        target = self._max_samples or 40_000
        # Each chunk holds ~2000 PLY files; add 2 extra for safety
        needed = max(1, (target // self._PLY_PER_CHUNK) + 2)
        # Cap at the actual number of chunks in the repo (109)
        needed = min(needed, 109)
        return needed

    @staticmethod
    def _ensure_pcd_extracted(repo_root: Path) -> None:
        """Extract PCD ZIP chunks if PLY files aren't already on disk."""
        import zipfile

        pcd_dir = repo_root / "pcd"
        if not pcd_dir.is_dir():
            return

        # Already extracted? (any .ply file present)
        try:
            for entry in pcd_dir.iterdir():
                if entry.suffix == ".ply":
                    return
        except OSError:
            return

        # Extract ZIP chunks
        zip_files = sorted(pcd_dir.glob("chunk_*.zip"))
        if not zip_files:
            log.warning("No PCD ZIP chunks found in %s", pcd_dir)
            return

        log.info("Extracting %d PCD ZIP chunks to %s ...", len(zip_files), pcd_dir)
        for zip_path in zip_files:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(pcd_dir)

        ply_count = sum(1 for _ in pcd_dir.glob("*.ply"))
        log.info("Extracted %d PLY files to %s", ply_count, pcd_dir)

    def _find_local_repo_root(self) -> Path | None:
        if self._data_root is None:
            return None
        candidate = Path(self._data_root) / "spatial" / "raw"
        if (candidate / "split.csv").is_file():
            return candidate
        return None

    def _load_local_metadata(self, repo_root: Path) -> list[dict[str, Any]]:
        split_csv = repo_root / "split.csv"
        with split_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    def _extract_from_dataset(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract point clouds + labels from a HuggingFace Dataset object.

        Parameters
        ----------
        ds : datasets.Dataset or datasets.DatasetDict
            Loaded dataset (from local disk or HuggingFace).

        Returns
        -------
        clouds : np.ndarray
            Shape ``(N_total, N_pts, 3)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        if hasattr(ds, "values"):
            from datasets import concatenate_datasets
            ds = concatenate_datasets(list(ds.values()))

        clouds_list: list[np.ndarray] = []
        labels_list: list[int] = []

        total = min(len(ds), self._max_samples) if self._max_samples else len(ds)
        for record in tqdm(ds, desc="Extracting SpatialLM", total=total):
            if self._max_samples is not None and len(clouds_list) >= self._max_samples:
                break

            cloud = self._extract_cloud(record)
            label = self._extract_label(record)

            if cloud is not None and label >= 0:
                clouds_list.append(cloud)
                labels_list.append(label)

        if not clouds_list:
            columns = getattr(ds, "column_names", None)
            if columns is None and len(ds) > 0 and isinstance(ds[0], dict):
                columns = list(ds[0].keys())
            raise RuntimeError(self._spatial_extraction_error(columns))

        clouds = np.stack(clouds_list, axis=0).astype(np.float32)
        labels = np.asarray(labels_list, dtype=np.int64)

        log.info(
            "SpatialAdapter: loaded %d clouds, shape=%s, classes=%d",
            clouds.shape[0],
            clouds.shape,
            len(np.unique(labels)),
        )
        return clouds, labels

    def _resolved_spec_from_sequences(self, sequences: np.ndarray) -> DatasetSpec:
        actual_dim = int(sequences.shape[-1])
        if actual_dim == self._spec.input_dim:
            return self._spec
        return replace(self._spec, input_dim=actual_dim)

    # ------------------------------------------------------------------ #

    def _extract_cloud(self, record: dict) -> np.ndarray | None:
        """Extract a (N, 3) point cloud from a HuggingFace record."""
        for key in ("points", "cloud", "xyz", "vertices", "coordinates"):
            if key in record:
                arr = self._coerce_cloud_array(record[key])
                if arr is not None:
                    return arr
        sample = record.get("sample")
        if isinstance(sample, dict):
            for key in ("points", "cloud", "xyz", "vertices", "coordinates"):
                if key in sample:
                    arr = self._coerce_cloud_array(sample[key])
                    if arr is not None:
                        return arr
        ply_path = self._resolve_ply_path(record)
        if ply_path is not None and ply_path.is_file():
            return self._load_ply_xyz(ply_path)
        return None

    @staticmethod
    def _coerce_cloud_array(value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] >= 3:
            return value[:, :3].astype(np.float32, copy=False)
        if isinstance(value, list):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]
        return None

    def _resolve_ply_path(self, record: dict[str, Any]) -> Path | None:
        """Resolve the local PLY file path for a record.

        Requires ``_raw_repo_root`` to be set (via local checkout or
        ``snapshot_download``).  Returns ``None`` if the file cannot
        be located locally — individual per-file download is not
        supported because SpatialLM stores PLYs inside ZIP chunks.
        """
        if self._raw_repo_root is None:
            return None

        path_value = record.get("pcd")
        if isinstance(path_value, str) and path_value:
            return self._raw_repo_root / path_value

        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            return None

        return self._raw_repo_root / "pcd" / f"{record_id}.ply"

    @staticmethod
    def _load_ply_xyz(path: Path) -> np.ndarray | None:
        try:
            with path.open("rb") as handle:
                fmt, vertex_count, vertex_props = SpatialAdapter._read_ply_header(handle)
                xyz_names = ("x", "y", "z")
                if not all(name in [prop_name for prop_name, _ in vertex_props] for name in xyz_names):
                    return None
                if fmt == "ascii":
                    return SpatialAdapter._read_ascii_vertices(handle, vertex_count, vertex_props)
                if fmt in {"binary_little_endian", "binary_big_endian"}:
                    return SpatialAdapter._read_binary_vertices(handle, fmt, vertex_count, vertex_props)
        except OSError:
            return None
        return None

    @staticmethod
    def _read_ply_header(handle) -> tuple[str, int, list[tuple[str, str]]]:
        format_name: str | None = None
        vertex_count = 0
        vertex_props: list[tuple[str, str]] = []
        current_element: str | None = None

        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            text = line.decode("ascii", errors="ignore").strip()
            if text == "end_header":
                break
            if text.startswith("format "):
                format_name = text.split()[1]
                continue
            if text.startswith("element "):
                parts = text.split()
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                    vertex_props = []
                continue
            if text.startswith("property ") and current_element == "vertex":
                parts = text.split()
                if len(parts) >= 3 and parts[1] != "list":
                    vertex_props.append((parts[2], parts[1]))

        if format_name is None or vertex_count <= 0 or not vertex_props:
            raise ValueError("Unsupported or empty PLY header")
        return format_name, vertex_count, vertex_props

    @staticmethod
    def _read_ascii_vertices(
        handle,
        vertex_count: int,
        vertex_props: list[tuple[str, str]],
    ) -> np.ndarray | None:
        prop_names = [name for name, _ in vertex_props]
        xyz_idx = [prop_names.index(axis) for axis in ("x", "y", "z")]
        points = np.empty((vertex_count, 3), dtype=np.float32)
        for idx in range(vertex_count):
            line = handle.readline()
            if not line:
                return None
            parts = line.decode("ascii", errors="ignore").strip().split()
            if len(parts) < len(prop_names):
                return None
            try:
                points[idx] = [float(parts[col]) for col in xyz_idx]
            except ValueError:
                return None
        return points

    @staticmethod
    def _read_binary_vertices(
        handle,
        fmt: str,
        vertex_count: int,
        vertex_props: list[tuple[str, str]],
    ) -> np.ndarray | None:
        type_map = {
            "char": "i1",
            "int8": "i1",
            "uchar": "u1",
            "uint8": "u1",
            "short": "i2",
            "int16": "i2",
            "ushort": "u2",
            "uint16": "u2",
            "int": "i4",
            "int32": "i4",
            "uint": "u4",
            "uint32": "u4",
            "float": "f4",
            "float32": "f4",
            "double": "f8",
            "float64": "f8",
        }
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype_fields: list[tuple[str, str]] = []
        for name, prop_type in vertex_props:
            dtype_code = type_map.get(prop_type.lower())
            if dtype_code is None:
                return None
            dtype_fields.append((name, f"{endian}{dtype_code}"))

        data = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=vertex_count)
        if data.shape[0] != vertex_count:
            return None
        try:
            points = np.stack([data["x"], data["y"], data["z"]], axis=1)
        except ValueError:
            return None
        return points.astype(np.float32, copy=False)

    def _spatial_extraction_error(self, columns: Any) -> str:
        return (
            "No valid point clouds extracted from SpatialLM. "
            f"Dataset columns: {columns}. "
            "The metadata table alone is not enough; SpatialLM stores coordinates in "
            "PCD/PLY files under pcd/{id}.ply. Make sure either "
            f"`{Path(self._data_root or 'data/datasets') / 'spatial' / 'raw'}` contains "
            "`split.csv` plus the `pcd/` directory, or allow the adapter to download "
            "the referenced `.ply` files from Hugging Face."
        )

    # Structured3D room-type taxonomy (source of SpatialLM-Dataset).
    # Maps string room_type → integer label.  "undefined" → -1 (excluded).
    _ROOM_TYPE_MAP: dict[str, int] = {
        "living room": 0,
        "kitchen": 1,
        "bedroom": 2,
        "bathroom": 3,
        "balcony": 4,
        "corridor": 5,
        "dining room": 6,
        "study": 7,
        "studio": 8,
        "store room": 9,
        "garden": 10,
        "laundry room": 11,
        "office": 12,
        "basement": 13,
        "garage": 14,
        "undefined": -1,
    }
    _ROOM_TYPE_ALIASES: dict[str, str] = {
        "livingroom": "living room",
        "living_room": "living room",
        "family room": "living room",
        "game room": "living room",
        "media room": "living room",
        "entertainment room": "living room",
        "diningroom": "dining room",
        "dining_room": "dining room",
        "storeroom": "store room",
        "store_room": "store room",
        "walk in closet": "store room",
        "closet combination": "store room",
        "closet": "store room",
        "furniture store": "store room",
        "clothing store": "store room",
        "laundryroom": "laundry room",
        "laundry_room": "laundry room",
        "restroom": "bathroom",
        "washroom": "bathroom",
        "primary bathroom": "bathroom",
        "guest bathroom": "bathroom",
        "hallway": "corridor",
        "entryway": "corridor",
        "foyer": "corridor",
        "mudroom": "corridor",
        "guestroom": "bedroom",
        "guest_room": "bedroom",
        "masterbedroom": "bedroom",
        "master_bedroom": "bedroom",
        "primary bedroom": "bedroom",
        "secondary bedroom": "bedroom",
        "child room": "bedroom",
        "childrens bedroom": "bedroom",
        "kidsroom": "bedroom",
        "kids_room": "bedroom",
        "study room": "study",
        "library": "study",
        "pantry": "kitchen",
        "home office": "office",
        "meeting room": "office",
        "home gym": "studio",
        "gym": "studio",
        "tatami room": "studio",
        "music room": "studio",
    }
    _warned_label_values: set[str] = set()
    _warned_missing_label: bool = False

    @classmethod
    def _normalize_room_type(cls, value: str) -> str:
        normalized = (
            value.strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
            .replace("'", "")
        )
        normalized = " ".join(normalized.split())
        compact = normalized.replace(" ", "")
        alias = cls._ROOM_TYPE_ALIASES.get(normalized) or cls._ROOM_TYPE_ALIASES.get(compact)
        return alias or normalized

    @classmethod
    def _infer_room_type(cls, normalized: str) -> str | None:
        primary = normalized.split(" with ", 1)[0].split(" combination", 1)[0].strip()
        if primary and primary != normalized:
            primary = cls._normalize_room_type(primary)
            if primary in cls._ROOM_TYPE_MAP:
                return primary
            inferred_primary = cls._infer_room_type(primary)
            if inferred_primary is not None:
                return inferred_primary

        tokens = normalized.split()
        joined = f" {normalized} "

        if "undefined" in tokens or normalized in {"other", "empty room"}:
            return "undefined"
        if "bathroom" in tokens or "spa" in tokens:
            return "bathroom"
        if "bedroom" in tokens:
            return "bedroom"
        if "kitchen" in tokens or "pantry" in tokens:
            return "kitchen"
        if "laundry" in tokens:
            return "laundry room"
        if "garage" in tokens:
            return "garage"
        if "basement" in tokens:
            return "basement"
        if "balcony" in tokens:
            return "balcony"
        if "garden" in tokens:
            return "garden"
        if any(term in joined for term in (" corridor ", " hallway ", " foyer ", " entryway ", " mudroom ")):
            return "corridor"
        if any(term in joined for term in (" dining ",)):
            return "dining room"
        if any(term in joined for term in (" living room ", " living area ", " family room ", " media room ", " game room ", " entertainment room ")):
            return "living room"
        if any(term in joined for term in (" study ", " study room ", " library ", " meditation room ")):
            return "study"
        if any(term in joined for term in (" office ", " home office ", " meeting room ")):
            return "office"
        if any(term in joined for term in (" closet ", " store ", " storage ")):
            return "store room"
        if any(term in joined for term in (" studio ", " gym ", " tatami room ", " music room ")):
            return "studio"
        return None

    @classmethod
    def _extract_label(cls, record: dict) -> int:
        """Extract an integer label from a HuggingFace record.

        For SpatialLM the label comes from the ``room_type`` field,
        which is a string (e.g. "bedroom").  We map it via the
        Structured3D room-type taxonomy.  Numeric labels are passed
        through directly.
        """
        for key in ("label", "room_type", "category", "class"):
            if key in record:
                val = record[key]
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="ignore")
                if isinstance(val, int):
                    return val
                if isinstance(val, (np.integer,)):
                    return int(val)
                if isinstance(val, str):
                    normalized = cls._normalize_room_type(val)
                    canonical = normalized
                    mapped = cls._ROOM_TYPE_MAP.get(canonical)
                    if mapped is None:
                        inferred = cls._infer_room_type(normalized)
                        if inferred is not None:
                            canonical = inferred
                            mapped = cls._ROOM_TYPE_MAP.get(canonical)
                    if mapped is not None:
                        return mapped
                    if val not in cls._warned_label_values:
                        cls._warned_label_values.add(val)
                        log.warning(
                            "Unrecognized SpatialLM room_type=%r (keys: %s); defaulting to 0",
                            val,
                            list(record.keys()),
                        )
                    return 0
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
        if not cls._warned_missing_label:
            cls._warned_missing_label = True
            log.warning(
                "No usable label column found in SpatialLM records (keys: %s); defaulting to 0",
                list(record.keys()),
            )
        return 0


register_adapter("spatial", SpatialAdapter)


__all__ = ["SpatialAdapter"]
