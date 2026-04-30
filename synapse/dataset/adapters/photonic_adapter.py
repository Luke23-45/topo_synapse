"""2D Photonic Topology Adapter.

Loads the 2D photonic topology dataset from HuggingFace and produces
a ``DatasetBundle`` for the Z3 scientific evaluation track.

The dataset contains 10,000 photonic crystal unit cells with labels
corresponding to topological symmetry settings and dielectric contrasts.
This is a direct test of the **Hodge Laplacian Proxy** — in photonics,
topology determines the existence of edge states.

Source: https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3, Track 3
- Data source doc: ``docs/dev/data_source.md`` §4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..preprocess.scientific import ScientificPreprocessor
from ..registry import register_adapter
from .base import DatasetBundle, DatasetSpec, Z3Adapter
from .persistence import get_prepared_cache_dir, load_prepared_bundle, save_prepared_bundle
from .split_utils import apply_split, extract_predefined_splits, try_load_from_local

log = logging.getLogger(__name__)


class PhotonicAdapter(Z3Adapter):
    """Adapter for the 2D photonic topology dataset.

    Parameters
    ----------
    inject_coordinates : bool
        If ``True``, prepend (x, y) grid coordinates to each feature
        vector, increasing ``input_dim`` by 2.
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
        inject_coordinates: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: int | None = None,
        data_root: str | None = None,
    ) -> None:
        self._inject_coordinates = inject_coordinates
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        self._max_samples = max_samples
        self._data_root = data_root

        # input_dim is 8 (features) + 2 (coordinates) if injected.
        effective_dim = 8 + (2 if inject_coordinates else 0)

        self._spec = DatasetSpec(
            name="photonic",
            modality="scientific",
            source="huggingface",
            hf_repo="cgeorgiaw/2d-photonic-topology",
            input_dim=effective_dim,
            sequence_length=64,
            num_classes=4,
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
        """Load 2D photonic topology and return splits (with disk caching).
        
        This is the primary entry point. It checks for prepared split
        files on disk before falling back to raw extraction.

        Returns
        -------
        DatasetBundle
        """
        # 1. Check for final prepared splits first
        cache_dir = Path(self._data_root or "data/datasets")
        cache_path = get_prepared_cache_dir(
            cache_dir, "photonic", self._spec, self._seed, self._train_ratio, self._val_ratio
        )
        
        bundle = load_prepared_bundle(cache_path, self._spec)
        if bundle is not None:
            return bundle

        # 2. If no prepared cache, load raw data (local or HF)
        log.info("No prepared cache found for PhotonicTopology. Starting extraction...")
        ds = self._load_data()
        split_result = extract_predefined_splits(
            ds,
            extract_array=self._extract_grid,
            extract_label=self._extract_label,
            max_samples=self._max_samples,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
        )
        if split_result is not None:
            grids, labels = split_result
        else:
            raw_grids, raw_labels = self._extract_from_dataset(ds)
            grids, labels = apply_split(
                raw_grids,
                raw_labels,
                train_ratio=self._train_ratio,
                val_ratio=self._val_ratio,
                seed=self._seed,
            )

        # 4. Wrap and Preprocess
        preprocessor = ScientificPreprocessor(
            inject_coordinates=self._inject_coordinates,
        )
        bundle = DatasetBundle(
            train_sequences=grids["train"],
            train_labels=labels["train"],
            val_sequences=grids["val"],
            val_labels=labels["val"],
            test_sequences=grids["test"],
            test_labels=labels["test"],
            spec=self._spec,
        )
        bundle = preprocessor(bundle)

        # 5. Save the FINAL PREPARED data to disk for next time
        save_prepared_bundle(cache_path, bundle)
        
        return bundle

    # ------------------------------------------------------------------ #

    def _load_data(self):
        """Load the raw photonic dataset object."""
        # --- Try local disk first ---
        local_ds = try_load_from_local(self._data_root, "photonic")
        if local_ds is not None:
            log.info("PhotonicAdapter: loading from local data_root=%s", self._data_root)
            return local_ds

        # --- Fall back to HuggingFace ---
        return self._load_from_huggingface()

    def _load_from_huggingface(self):
        """Download .jld2 files from HuggingFace and read with h5py.

        The cgeorgiaw/2d-photonic-topology repo stores Julia JLD2 files
        (HDF5-compatible), not standard Parquet/CSV.  The ``datasets``
        library cannot auto-detect these, so we download via
        ``huggingface_hub`` and read with ``h5py``.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "The 'h5py' package is required to read .jld2 photonic "
                "topology files. Install with: pip install h5py"
            ) from exc

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_hub' package is required to download "
                "the photonic topology dataset. Install with: "
                "pip install huggingface_hub"
            ) from exc

        repo = self._spec.hf_repo
        log.info("Loading 2D photonic topology from HuggingFace: %s", repo)

        # Download only the lattices config (smallest, ~30 MB per file).
        # Each lattices-planegroup$num.jld2 contains flatv, isovalv, Rsv.
        local_dir = snapshot_download(
            repo,
            repo_type="dataset",
            allow_patterns=["lattices/*"],
        )
        log.info("Downloaded photonic lattices to: %s", local_dir)

        return self._read_jld2_lattices(local_dir)

    # ---- JLD2 / HDF5 reading --------------------------------------------- #

    # Plane group → crystal system mapping (4 classes matching num_classes=4).
    _PG_TO_CLASS: dict[int, int] = {
        **{pg: 0 for pg in range(1, 3)},     # oblique
        **{pg: 1 for pg in range(3, 10)},    # rectangular
        **{pg: 2 for pg in range(10, 13)},   # square
        **{pg: 3 for pg in range(13, 18)},   # hexagonal
    }

    def _read_jld2_lattices(self, local_dir: str) -> list[dict[str, Any]]:
        """Read .jld2 lattice files and return a list of record dicts.

        Each JLD2 file (HDF5-compatible) contains three 10 000-element
        vectors — ``flatv``, ``isovalv``, ``Rsv`` — for a given plane
        group.  We extract numerical data via ``h5py``, build (H, W, F)
        grids, and assign a crystal-system label (4 classes).
        """
        import h5py
        import re

        base = Path(local_dir)
        jld2_files = sorted(base.glob("**/lattices-planegroup*.jld2"))

        if not jld2_files:
            raise FileNotFoundError(
                f"No lattices JLD2 files found in {local_dir}"
            )

        all_records: list[dict[str, Any]] = []
        for fpath in tqdm(jld2_files, desc="Reading photonic JLD2"):
            match = re.search(r"planegroup(\d+)", fpath.name)
            pg_num = int(match.group(1)) if match else 0
            label = self._PG_TO_CLASS.get(pg_num, pg_num % 4)

            with h5py.File(str(fpath), "r") as f:
                records = self._parse_jld2_lattice_file(f, label)
                all_records.extend(records)

            if self._max_samples and len(all_records) >= self._max_samples:
                break

        log.info(
            "Read %d photonic records from %d JLD2 files",
            len(all_records), len(jld2_files),
        )
        return all_records

    def _parse_jld2_lattice_file(
        self, h5file: Any, label: int
    ) -> list[dict[str, Any]]:
        """Parse a single JLD2 lattice file into record dicts."""
        import h5py

        # --- isovalv (isovalues) — simple float vector ----------------------
        isovalv: np.ndarray | None = None
        if "isovalv" in h5file:
            ds = h5file["isovalv"]
            if isinstance(ds, h5py.Dataset):
                isovalv = np.asarray(ds, dtype=np.float32).ravel()

        # --- Rsv (lattice vectors) — (N, 2, 2) or similar ------------------
        Rsv: np.ndarray | None = None
        if "Rsv" in h5file:
            ds = h5file["Rsv"]
            if isinstance(ds, h5py.Dataset):
                Rsv = np.asarray(ds, dtype=np.float32)
                if Rsv.ndim == 1:
                    Rsv = Rsv.reshape(-1, 1)
                elif Rsv.ndim > 2:
                    Rsv = Rsv.reshape(Rsv.shape[0], -1)

        # --- flatv (Fourier lattices) — vector of custom Julia objects ------
        fourier_grids: np.ndarray | None = None
        if "flatv" in h5file:
            fourier_grids = self._eval_fourier_from_hdf5(
                h5file["flatv"], isovalv, grid_res=8,
            )

        n_samples = (
            len(fourier_grids) if fourier_grids is not None
            else len(isovalv) if isovalv is not None
            else 0
        )
        if n_samples == 0:
            log.warning(
                "No samples in JLD2 file (keys: %s)", list(h5file.keys())
            )
            return []

        records: list[dict[str, Any]] = []
        for i in range(n_samples):
            if self._max_samples and len(records) >= self._max_samples:
                break

            if fourier_grids is not None:
                grid = fourier_grids[i]
            else:
                grid = self._build_fallback_grid(isovalv, Rsv, i)

            if grid is not None:
                records.append({"grid": grid, "label": label})

        return records

    # ------------------------------------------------------------------ #

    def _eval_fourier_from_hdf5(
        self, flatv_node: Any, isovalv: np.ndarray | None, grid_res: int = 8,
    ) -> np.ndarray | None:
        """Try to evaluate Fourier lattices on a 2D grid.

        Returns ``(N, grid_res, grid_res, F)`` or ``None``.
        """
        import h5py

        # Simple dataset — use directly if it's a numeric array
        if isinstance(flatv_node, h5py.Dataset):
            data = np.asarray(flatv_node, dtype=np.float32)
            if data.ndim >= 2:
                return data
            return None

        if not isinstance(flatv_node, h5py.Group):
            return None

        # Group of numbered sub-groups (one per sample)
        digit_keys = sorted(
            [k for k in flatv_node.keys() if k.isdigit()],
            key=int,
        )
        if not digit_keys:
            return None

        # Collect per-sample numerical vectors
        per_sample: list[np.ndarray] = []
        for key in digit_keys:
            vec = self._collect_leaf_arrays(flatv_node[key])
            if vec is not None and vec.size > 0:
                per_sample.append(vec)

        if not per_sample:
            return None

        # Uniform-length coefficient matrix (N, L)
        max_len = max(v.size for v in per_sample)
        N = len(per_sample)
        coeffs = np.zeros((N, max_len), dtype=np.float32)
        for i, v in enumerate(per_sample):
            coeffs[i, :v.size] = v

        # Attempt Fourier-series evaluation on a grid
        grids = self._fourier_coeffs_to_grids(coeffs, isovalv, grid_res)
        if grids is not None:
            return grids

        # Fallback: reshape coefficient vectors into grid form
        return self._coeffs_to_grid_fallback(coeffs, grid_res)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_leaf_arrays(node: Any) -> np.ndarray | None:
        """Recursively gather all leaf numeric datasets in an HDF5 node."""
        import h5py

        parts: list[np.ndarray] = []

        if isinstance(node, h5py.Dataset):
            try:
                return np.asarray(node, dtype=np.float32).ravel()
            except Exception:
                return None

        if isinstance(node, h5py.Group):
            for key in sorted(node.keys()):
                child = node[key]
                arr = PhotonicAdapter._collect_leaf_arrays(child)
                if arr is not None and arr.size > 0:
                    parts.append(arr)

        if not parts:
            return None
        return np.concatenate(parts)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _fourier_coeffs_to_grids(
        coeffs: np.ndarray, isovalv: np.ndarray | None, grid_res: int,
    ) -> np.ndarray | None:
        """Evaluate Fourier coefficients on a 2D grid.

        Interprets the first half of each coefficient vector as real parts
        and the second half as imaginary parts of complex Fourier amplitudes.
        Returns ``(N, grid_res, grid_res, 2)`` — [level-set, dielectric].
        """
        N, L = coeffs.shape
        n_complex = L // 2
        if n_complex < 4:
            return None

        r1 = np.linspace(0, 1, grid_res, dtype=np.float32)
        r2 = np.linspace(0, 1, grid_res, dtype=np.float32)
        R1, R2 = np.meshgrid(r1, r2)

        n_side = int(np.ceil(np.sqrt(n_complex)))
        m_range = np.arange(-n_side // 2 + 1, n_side // 2 + 1)
        n_range = m_range  # same range for both axes

        grids = np.zeros((N, grid_res, grid_res, 2), dtype=np.float32)

        for i in range(N):
            real_c = coeffs[i, :n_complex]
            imag_c = coeffs[i, n_complex : 2 * n_complex]
            complex_c = real_c + 1j * imag_c

            level_set = np.zeros((grid_res, grid_res), dtype=np.float64)
            idx = 0
            for m in m_range:
                for n in n_range:
                    if idx >= n_complex:
                        break
                    level_set += np.real(
                        complex_c[idx]
                        * np.exp(2j * np.pi * (m * R1 + n * R2))
                    ).astype(np.float64)
                    idx += 1
                if idx >= n_complex:
                    break

            ls32 = level_set.astype(np.float32)
            grids[i, :, :, 0] = ls32
            if isovalv is not None and i < len(isovalv):
                grids[i, :, :, 1] = (ls32 > isovalv[i]).astype(np.float32)
            else:
                grids[i, :, :, 1] = (ls32 > 0).astype(np.float32)

        return grids

    # ------------------------------------------------------------------ #

    @staticmethod
    def _coeffs_to_grid_fallback(
        coeffs: np.ndarray, grid_res: int,
    ) -> np.ndarray:
        """Reshape coefficient vectors into a grid as a last resort."""
        N, L = coeffs.shape
        hw = grid_res * grid_res
        F = max(1, L // hw)
        used = hw * F
        result = np.zeros((N, grid_res, grid_res, F), dtype=np.float32)
        for i in range(N):
            chunk = coeffs[i, :used]
            if chunk.size < used:
                chunk = np.pad(chunk, (0, used - chunk.size))
            result[i] = chunk.reshape(grid_res, grid_res, F)
        return result

    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_fallback_grid(
        isovalv: np.ndarray | None,
        Rsv: np.ndarray | None,
        idx: int,
    ) -> np.ndarray | None:
        """Build a minimal (8, 8, 8) grid from isovalue + lattice vectors."""
        feats: list[float] = []
        if isovalv is not None and idx < len(isovalv):
            feats.append(float(isovalv[idx]))
        if Rsv is not None and idx < len(Rsv):
            feats.extend(Rsv[idx].ravel().tolist())

        if not feats:
            return None

        arr = np.array(feats, dtype=np.float32)
        if arr.size < 8:
            arr = np.pad(arr, (0, 8 - arr.size))
        else:
            arr = arr[:8]

        return np.broadcast_to(arr, (8, 8, 8)).copy()

    # ------------------------------------------------------------------ #

    def _extract_from_dataset(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract grids + labels from a HuggingFace Dataset or record list.

        Parameters
        ----------
        ds : datasets.Dataset, datasets.DatasetDict, or list[dict]
            Loaded dataset (from local disk, HuggingFace, or JLD2 reader).

        Returns
        -------
        grids : np.ndarray
            Shape ``(N_total, H, W, F)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        if hasattr(ds, "values") and not isinstance(ds, dict):
            from datasets import concatenate_datasets
            ds = concatenate_datasets(list(ds.values()))

        grids_list: list[np.ndarray] = []
        labels_list: list[int] = []

        total = min(len(ds), self._max_samples) if self._max_samples else len(ds)
        for record in tqdm(ds, desc="Extracting PhotonicTopology", total=total):
            if self._max_samples is not None and len(grids_list) >= self._max_samples:
                break

            grid = self._extract_grid(record)
            label = self._extract_label(record)

            if grid is not None:
                grids_list.append(grid)
                labels_list.append(label)

        if not grids_list:
            cols = getattr(ds, "column_names", list(ds[0].keys()) if ds else [])
            raise RuntimeError(
                f"No valid grids extracted from photonic topology. "
                f"Dataset columns: {cols}"
            )

        grids = np.stack(grids_list, axis=0).astype(np.float32)
        labels = np.asarray(labels_list, dtype=np.int64)

        log.info(
            "PhotonicAdapter: loaded %d grids, shape=%s, classes=%d",
            grids.shape[0],
            grids.shape,
            len(np.unique(labels)),
        )
        return grids, labels


    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_grid(record: dict) -> np.ndarray | None:
        """Extract a (H, W, F) grid from a HuggingFace record."""
        for key in ("grid", "unit_cell", "image", "features", "data"):
            if key in record:
                val = record[key]
                if isinstance(val, np.ndarray) and val.ndim in (2, 3):
                    if val.ndim == 2:
                        val = val[:, :, np.newaxis]
                    return val
                if isinstance(val, list):
                    arr = np.asarray(val, dtype=np.float32)
                    if arr.ndim in (2, 3):
                        if arr.ndim == 2:
                            arr = arr[:, :, np.newaxis]
                        return arr
        return None

    @staticmethod
    def _extract_label(record: dict) -> int:
        """Extract an integer label from a HuggingFace record."""
        for key in ("label", "topology", "symmetry", "class", "category"):
            if key in record:
                val = record[key]
                if isinstance(val, int):
                    return val
                if isinstance(val, (np.integer,)):
                    return int(val)
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
        log.warning(
            "No label column found in record (keys: %s), defaulting to 0",
            list(record.keys()),
        )
        return 0


register_adapter("photonic", PhotonicAdapter)


__all__ = ["PhotonicAdapter"]
