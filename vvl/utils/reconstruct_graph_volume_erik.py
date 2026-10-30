"""
Reconstruct a filtered vessel segmentation from a saved VesselVio pickle.

This script uses ``vvl.analysis.reconstruct_volume``.  That function does not
rasterize graph edges or radii; it filters the corresponding source
segmentation according to the skeleton positions retained by the graph.
Consequently, both the VesselVio ``.pkl`` and its source segmentation voxel
array are required.

The input NIfTI's spatial metadata is deliberately ignored.  Output geometry
is inferred from the configured extraction resolution using a zero-origin,
axis-aligned affine.

Configuration is kept at the bottom for direct execution in VS Code.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import nibabel as nib
import numpy as np

from vvl.analysis import reconstruct_volume as vvl_reconstruct_volume
from vvl.utils.volume_processing import pad_volume, skeletonize, volume_prep


@dataclass(frozen=True)
class ReconstructionConfig:
    """
    Settings that must match the original VesselVio graph extraction.

    ``resolution`` follows the array-axis order used by VesselVio during
    extraction.  It is not read from the input NIfTI header.

    Set ``legacy_swap_axes`` to the same ``legacy`` value used by GraphInfo.
    Legacy extraction swaps array axes 0 and 2 before graph construction.
    """

    resolution: Tuple[float, float, float] = (0.012, 0.012, 0.003)
    legacy_swap_axes: bool = False
    strict: bool = False

    def __post_init__(self) -> None:
        resolution = np.asarray(self.resolution, dtype=float)
        if (
            resolution.shape != (3,)
            or np.any(~np.isfinite(resolution))
            or np.any(resolution <= 0)
        ):
            raise ValueError(
                "resolution must contain three finite positive values"
            )


@dataclass(frozen=True)
class ReconstructionStats:
    input_foreground_voxels: int
    output_foreground_voxels: int
    retained_fraction: float
    output_foreground_volume: float
    n_graph_vertices: int
    n_graph_edges: int


def load_vesselvio_pickle(path: Path) -> Any:
    """
    Load a trusted VesselVio igraph pickle.

    Python pickle can execute code while loading and must not be used with
    untrusted files.
    """
    path = Path(path)
    with path.open("rb") as file:
        graph = pickle.load(file)

    if not hasattr(graph, "vs") or not hasattr(graph, "es"):
        raise TypeError(
            f"Expected a VesselVio igraph in {path}, got {type(graph)!r}"
        )
    if "original_edge_positions" not in graph.es.attribute_names():
        raise ValueError(
            "The pickle lacks 'original_edge_positions'. "
            "Use the full VesselVio .pkl saved before GraphML attribute removal."
        )
    return graph


def load_segmentation_voxels(path: Path) -> np.ndarray:
    """Load only the binary voxel array; ignore all source NIfTI metadata."""
    image = nib.load(Path(path))
    volume = np.asanyarray(image.dataobj)
    if volume.ndim != 3:
        raise ValueError(
            f"Source segmentation must be 3-D, got shape {volume.shape}"
        )
    if not np.any(volume):
        raise ValueError("Source segmentation contains no foreground voxels")
    return np.asarray(volume > 0, dtype=np.uint8)


def _restore_padded_graph_positions(graph: Any) -> Any:
    """
    Restore the one-voxel padding present when the graph was constructed.

    ``extract_graph_from_volume`` subtracts one from graph coordinates before
    saving.  ``reconstruct_volume`` is called on the padded working volume, so
    saved original edge positions must be shifted back by one temporarily.
    """
    restored = graph.copy()
    positions = restored.es["original_edge_positions"]
    restored.es["original_edge_positions"] = [
        np.asarray(edge_positions, dtype=int) + 1
        for edge_positions in positions
    ]
    return restored


def _restore_original_extent(
    bounded_mask: np.ndarray,
    point_minima: Sequence[int],
    point_maxima: Sequence[int],
) -> np.ndarray:
    if any(size < 3 for size in bounded_mask.shape):
        raise ValueError(
            "Padded reconstruction has an invalid shape: "
            f"{bounded_mask.shape}"
        )
    unpadded = bounded_mask[1:-1, 1:-1, 1:-1]
    padding = tuple(
        (int(before), int(after))
        for before, after in zip(point_minima, point_maxima)
    )
    return np.pad(unpadded, padding, mode="constant")


def reconstruct_from_pickle(
    graph: Any,
    source_segmentation: np.ndarray,
    *,
    config: ReconstructionConfig,
) -> Tuple[np.ndarray, ReconstructionStats]:
    """
    Filter a source segmentation using a saved reduced VesselVio graph.

    This reproduces the preprocessing around ``vvl.analysis.reconstruct_volume``
    and uses ``original_edge_positions`` because saved pickles contain reduced
    graphs rather than the original voxel-level graph.
    """
    source = np.asarray(source_segmentation > 0, dtype=np.uint8)
    if source.ndim != 3:
        raise ValueError(
            f"source_segmentation must be 3-D, got shape {source.shape}"
        )
    if not np.any(source):
        raise ValueError("source_segmentation contains no foreground voxels")

    working_source = (
        source.swapaxes(0, 2) if config.legacy_swap_axes else source.copy()
    )
    original_working_shape = working_source.shape

    bounded_volume, point_minima, point_maxima = volume_prep(working_source)
    padded_volume = pad_volume(bounded_volume)
    points = skeletonize(padded_volume)
    if len(points) == 0:
        raise ValueError("Skeletonization produced no centerline points")

    padded_graph = _restore_padded_graph_positions(graph)
    assigned = vvl_reconstruct_volume(
        padded_volume,
        padded_graph,
        points,
        np.asarray(config.resolution, dtype=float),
        point_minima,
        strict=config.strict,
        use_unreduced_nodes=True,
    )
    reconstructed_working = _restore_original_extent(
        assigned >= 0,
        point_minima,
        point_maxima,
    )
    if reconstructed_working.shape != original_working_shape:
        raise RuntimeError(
            "Reconstructed working volume has unexpected shape "
            f"{reconstructed_working.shape}; expected {original_working_shape}"
        )

    reconstructed = (
        reconstructed_working.swapaxes(0, 2)
        if config.legacy_swap_axes
        else reconstructed_working
    )
    reconstructed = np.asarray(reconstructed, dtype=np.uint8)
    if reconstructed.shape != source.shape:
        raise RuntimeError(
            f"Output shape {reconstructed.shape} differs from input {source.shape}"
        )

    input_voxels = int(np.count_nonzero(source))
    output_voxels = int(np.count_nonzero(reconstructed))
    voxel_volume = float(np.prod(config.resolution))
    stats = ReconstructionStats(
        input_foreground_voxels=input_voxels,
        output_foreground_voxels=output_voxels,
        retained_fraction=output_voxels / input_voxels,
        output_foreground_volume=output_voxels * voxel_volume,
        n_graph_vertices=graph.vcount(),
        n_graph_edges=graph.ecount(),
    )
    return reconstructed, stats


def inferred_output_resolution(
    config: ReconstructionConfig,
) -> Tuple[float, float, float]:
    """Return voxel sizes in the saved output array's axis order."""
    resolution = np.asarray(config.resolution, dtype=float)
    if config.legacy_swap_axes:
        resolution = resolution[[2, 1, 0]]
    return tuple(float(value) for value in resolution)


def save_inferred_nifti(
    output_path: Path,
    volume: np.ndarray,
    *,
    config: ReconstructionConfig,
) -> None:
    """
    Save uint8 NIfTI with inferred geometry and no copied source metadata.

    The affine has zero translation, no rotation/shear, and diagonal voxel
    sizes taken from ``config.resolution``.
    """
    output_path = Path(output_path)
    if not str(output_path).lower().endswith((".nii", ".nii.gz")):
        raise ValueError("output_path must end with .nii or .nii.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_resolution = inferred_output_resolution(config)
    affine = np.eye(4, dtype=float)
    affine[0, 0], affine[1, 1], affine[2, 2] = output_resolution

    image = nib.Nifti1Image(np.asarray(volume, dtype=np.uint8), affine)
    image.header.set_data_dtype(np.uint8)
    image.header.set_xyzt_units("mm")
    image.header["cal_min"] = 0
    image.header["cal_max"] = 1
    nib.save(image, output_path)


def reconstruct_pickle_to_nifti(
    pickle_path: Path,
    segmentation_path: Path,
    output_path: Path,
    *,
    config: ReconstructionConfig,
) -> ReconstructionStats:
    """Run the complete VesselVio reconstruction and save an inferred NIfTI."""
    graph = load_vesselvio_pickle(pickle_path)
    source = load_segmentation_voxels(segmentation_path)
    reconstructed, stats = reconstruct_from_pickle(
        graph,
        source,
        config=config,
    )
    save_inferred_nifti(output_path, reconstructed, config=config)

    print(f"VesselVio pickle:       {pickle_path}")
    print(f"Segmentation voxels:    {segmentation_path}")
    print(f"Output NIfTI:           {output_path}")
    print(f"Graph vertices:         {stats.n_graph_vertices}")
    print(f"Graph edges:            {stats.n_graph_edges}")
    print(f"Input foreground:       {stats.input_foreground_voxels} voxels")
    print(f"Output foreground:      {stats.output_foreground_voxels} voxels")
    print(f"Retained segmentation:  {stats.retained_fraction:.2%}")
    print(f"Output vessel volume:   {stats.output_foreground_volume:.6f} mm^3")
    print(f"Output voxel sizes:     {inferred_output_resolution(config)} mm")
    return stats


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

PICKLE_PATH = Path(r"C:\Users\erik\projects\hallucination_eval\data\i16\vessel_pred\vesselvio\R_20190306175255_Julia_BeforeCuffpoint_Base_3_RSOM50_wl1_corr_v_rgb_pred.pkl")
SEGMENTATION_PATH = Path(r"C:\Users\erik\projects\hallucination_eval\data\i16\vessel_pred\R_20190306175255_Julia_BeforeCuffpoint_Base_3_RSOM50_wl1_corr_v_rgb_pred.nii.gz")
OUTPUT_NIFTI_PATH = Path(r"C:\Users\erik\projects\hallucination_eval\data\i16\vessel_pred\vesselvio\R_20190306175255_Julia_BeforeCuffpoint_Base_3_RSOM50_wl1_corr_v_rgb_pred_reconstructed.nii.gz")

# These values must match the settings used for the original graph extraction.
CONFIG = ReconstructionConfig(
    resolution=(1, 1, 1),
    legacy_swap_axes=False,
    strict=False,
)


if __name__ == "__main__":
    reconstruct_pickle_to_nifti(
        PICKLE_PATH,
        SEGMENTATION_PATH,
        OUTPUT_NIFTI_PATH,
        config=CONFIG,
    )
