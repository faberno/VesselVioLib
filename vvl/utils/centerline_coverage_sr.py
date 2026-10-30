"""Centerline coverage of super-resolution vessel graphs against the original, like ``centerline_coverage_erik.py``.

Reference: graphs of the original recon, segmented on the common crop
(``original/vessel_segmentation_vesnet_common/vesselvio``). Predictions: graphs of the 9x / 16x / 25x SR
inference volumes (``<factor>/interpolated/OSRecon/vessel_segmentation_vesnet/vesselvio``). All graphs come
from ``extract_graphs_cropped_vesseg.py`` on masks of identical shape and origin (``rsom_vesseg_vesnet_sr.py``),
so their coordinates are directly comparable without any registration.

Two differences to the configuration in ``centerline_coverage_erik.py``:

* Axis order. The vessel masks are stored as (a, b, z), and ``save_graph`` writes X, Y, Z = axis 2, 1, 0,
  so graph X is depth here: ``SPACING_XYZ = (0.003, 0.012, 0.012)``. With this spacing the resampled
  polyline length equals VesselVio's summed edge ``length`` exactly.
* Radius. ``radii_list`` in VesselVio graphs holds the visualization radii in voxel units (see
  ``feature_processing.feature_extraction``, which stores ``vis_radii`` in that field), not physical radii,
  while ``radius_avg`` is in mm. ``centerline_coverage_erik`` prefers ``radii_list`` when present, which
  inflates the tolerance by ~100x. With ``RADIUS_SOURCE = "radius_avg"`` the list is dropped before the
  comparison, so tolerances use the per-edge physical mean radius.

Outputs in OUTPUT_DIR: ``centerline_coverage_summary.csv`` (one row per scan and factor),
``centerline_coverage_edges.csv`` (per edge) and ``centerline_coverage_pooled.csv`` (pooled per factor).
"""

from dataclasses import asdict
from pathlib import Path

import networkx as nx

from vvl.utils.centerline_coverage_erik import (
    RadiusTolerance,
    _write_csv,
    compare_graphs,
    load_graph,
    print_label_summary,
    summarize_by_label,
)

# ---------------------------------------------------------------------------- configuration
STUDY_DIR = Path(r"E:\sr_data\532\532_raw_PatrickJoshuaNew_forTesting\selection")
REFERENCE_DIR = STUDY_DIR / "original" / "vessel_segmentation_vesnet_common" / "vesselvio"
PREDICTION_DIRS = {
    f: STUDY_DIR / f / "interpolated" / "OSRecon" / "vessel_segmentation_vesnet" / "vesselvio"
    for f in ["9x", "16x", "25x"]
}
OUTPUT_DIR = STUDY_DIR / "centerline_coverage_vesnet"

SPACING_XYZ = (0.003, 0.012, 0.012)  # mm per graph unit in X (= z, depth), Y (= b), Z (= a)
RADIUS_SOURCE = "radius_avg"  # "radius_avg" (physical, per edge) or "radii_list" (VesselVio vis radii, voxels)

# Same sampling and tolerance as centerline_coverage_erik.py
SAMPLE_STEP = 0.003  # mm, matches the finest voxel axis
TOLERANCE = RadiusTolerance(radius_factor=2.0, minimum=0.006, maximum=None)
# ------------------------------------------------------------------------------------------


def load(path: Path) -> nx.Graph:
    graph = load_graph(path)
    if RADIUS_SOURCE == "radius_avg":
        for *_, data in graph.edges(data=True):
            data.pop("radii_list", None)
    elif RADIUS_SOURCE != "radii_list":
        raise ValueError(f"unknown RADIUS_SOURCE {RADIUS_SOURCE!r}")
    return graph


def main():
    reference_paths = sorted(REFERENCE_DIR.glob("*.pkl"))
    if not reference_paths:
        raise FileNotFoundError(f"No reference graphs in {REFERENCE_DIR}")
    print(f"{len(reference_paths)} reference graphs, factors {list(PREDICTION_DIRS)}")

    summary_rows, edge_rows = [], []
    for i, ref_path in enumerate(reference_paths, start=1):
        reference = load(ref_path)
        for label, pred_dir in PREDICTION_DIRS.items():
            pred_path = pred_dir / ref_path.name
            if not pred_path.is_file():
                print(f"  missing {label} graph for {ref_path.stem}, skipping")
                continue
            result = compare_graphs(
                reference, load(pred_path),
                reference_spacing_xyz=SPACING_XYZ, prediction_spacing_xyz=SPACING_XYZ,
                sample_step=SAMPLE_STEP, tolerance=TOLERANCE,
            )
            summary_rows.append({"sample": ref_path.stem, "label": label, **result.summary_dict()})
            edge_rows.extend({"sample": ref_path.stem, "label": label, **asdict(e)}
                             for e in (*result.reference_edges, *result.prediction_edges))
            print(f"[{i}/{len(reference_paths)}] {label:>4} recall={result.reference.fraction:.3f} "
                  f"precision={result.prediction.fraction:.3f} F1={result.f1:.3f} "
                  f"length bias={result.length_bias_percent:+.1f}%  {ref_path.stem[:40]}")

    _write_csv(OUTPUT_DIR / "centerline_coverage_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "centerline_coverage_edges.csv", edge_rows)
    _write_csv(OUTPUT_DIR / "centerline_coverage_pooled.csv", summarize_by_label(summary_rows, list(PREDICTION_DIRS)))
    print_label_summary(summary_rows, list(PREDICTION_DIRS))
    print(f"\nResults -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
