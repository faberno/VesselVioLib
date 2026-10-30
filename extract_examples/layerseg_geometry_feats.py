"""Extract epidermis geometry features from layer segmentations only.

Configure ``layer_segmentations_dir`` in the ``__main__`` block and run this
file directly.  Unlike ``intensity_feats.py``, this script neither discovers
nor loads reconstruction or vessel-segmentation files.
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DEPTH_VOXEL_SIZE_UM = 3.0


def get_surface_angle(surface: np.ndarray) -> int:
    """Return the integer tilt angle of the best-fit surface plane in degrees.

    This computes the same least-squares slopes as the explicit design matrix
    in ``intensity_feats.py`` without allocating three flattened coordinate
    arrays.  That substantially lowers peak memory use for large masks.
    """
    if surface.ndim != 2:
        raise ValueError(f"Expected a 2D surface, got shape {surface.shape}.")

    rows, columns = surface.shape
    surface_float = np.asarray(surface, dtype=np.float64)

    if columns > 1:
        x = np.arange(columns, dtype=np.float64)
        x -= x.mean()
        slope_x = np.sum(surface_float * x[None, :]) / (
            rows * np.sum(x * x)
        )
    else:
        slope_x = 0.0

    if rows > 1:
        y = np.arange(rows, dtype=np.float64)
        y -= y.mean()
        slope_y = np.sum(surface_float * y[:, None]) / (
            columns * np.sum(y * y)
        )
    else:
        slope_y = 0.0

    angle = np.degrees(np.arctan(np.hypot(slope_x, slope_y)))
    return int(angle)


def extract_layerseg_geometry_features(
    volume_lay: np.ndarray,
    depth_voxel_size_um: float = DEFAULT_DEPTH_VOXEL_SIZE_UM,
) -> Dict[str, int]:
    """Compute the three geometry features used by ``intensity_feats.py``.

    The last array axis is treated as depth.  Values greater than zero belong
    to the layer mask.  Integer conversion intentionally preserves the legacy
    truncation behaviour of the original feature extractor.
    """
    volume_lay = np.asarray(volume_lay)
    if volume_lay.ndim != 3:
        raise ValueError(
            f"Expected a 3D layer segmentation, got shape {volume_lay.shape}."
        )
    if depth_voxel_size_um <= 0:
        raise ValueError("depth_voxel_size_um must be greater than zero.")

    mask = volume_lay > 0
    depth_counts = np.count_nonzero(mask, axis=-1)
    populated_columns = depth_counts > 0
    if not np.any(populated_columns):
        raise ValueError("The layer segmentation is empty.")

    # Keep np.argmax(volume_lay), rather than argmax(mask), for exact
    # compatibility with the existing extractor on labelled segmentations.
    surface = np.argmax(volume_lay, axis=-1)
    layseg_thickness = np.mean(depth_counts[populated_columns])
    dist_transducer = np.mean(surface)

    return {
        "layseg_thickness [um]": int(layseg_thickness * depth_voxel_size_um),
        "dist_transducer [um]": int(dist_transducer * depth_voxel_size_um),
        "surface_angle": get_surface_angle(surface),
    }


def _nifti_stem(path: Path) -> str:
    """Strip either the .nii or .nii.gz suffix from a path."""
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _iter_layer_segmentations(directory: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.name.lower().endswith(".nii.gz") or path.suffix.lower() == ".nii")
    )


def _process_one(
    args: Tuple[Path, float],
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Load and process one segmentation; kept top-level for multiprocessing."""
    path, depth_voxel_size_um = args
    try:
        # asanyarray avoids get_fdata()'s unconditional float64 conversion.
        volume_lay = np.asanyarray(nib.load(path).dataobj)
        features = extract_layerseg_geometry_features(
            volume_lay,
            depth_voxel_size_um=depth_voxel_size_um,
        )
        return {"name": _nifti_stem(path), **features}, None
    except Exception as exc:
        return None, f"{path.name}: {exc!r}"


def extract_directory(
    layer_segmentations_dir: Path,
    results_folder: Path,
    depth_voxel_size_um: float = DEFAULT_DEPTH_VOXEL_SIZE_UM,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Extract all layer-segmentation geometry features and write CSV/XLSX."""
    layer_segmentations_dir = Path(layer_segmentations_dir)
    results_folder = Path(results_folder)
    if not layer_segmentations_dir.is_dir():
        raise NotADirectoryError(layer_segmentations_dir)
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    paths = list(_iter_layer_segmentations(layer_segmentations_dir))
    if not paths:
        raise FileNotFoundError(
            f"No .nii or .nii.gz files found in {layer_segmentations_dir}."
        )

    tasks = [(path, depth_voxel_size_um) for path in paths]
    if n_workers == 1:
        processed = map(_process_one, tasks)
        outputs = list(tqdm(processed, total=len(tasks), desc="Layer geometry"))
    else:
        with Pool(processes=n_workers) as pool:
            outputs = list(
                tqdm(
                    pool.imap_unordered(_process_one, tasks),
                    total=len(tasks),
                    desc="Layer geometry",
                )
            )

    failures = [error for _, error in outputs if error is not None]
    if failures:
        print(f"Failed to process {len(failures)} segmentation(s):")
        for failure in failures:
            print(f"  {failure}")

    results = [result for result, _ in outputs if result is not None]
    if not results:
        raise RuntimeError("No layer segmentations were processed successfully.")

    dataframe = pd.DataFrame(results).sort_values("name").reset_index(drop=True)
    results_folder.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(results_folder / "features_layerseg_geometry.csv", index=False)
    dataframe.to_excel(results_folder / "features_layerseg_geometry.xlsx", index=False)
    return dataframe


if __name__ == "__main__":
    # Configuration for direct execution in VS Code.
    layer_segmentations_dir = Path(
        r"E:\CVD_backup\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\epidermis\arm"
    )
    results_folder = layer_segmentations_dir / "layer_feats"

    depth_voxel_size_um = 3.0
    n_workers = min(8, max(1, cpu_count() - 1))

    result_table = extract_directory(
        layer_segmentations_dir=layer_segmentations_dir,
        results_folder=results_folder,
        depth_voxel_size_um=depth_voxel_size_um,
        n_workers=n_workers,
    )
    print(f"Wrote geometry features for {len(result_table)} segmentation(s).")
