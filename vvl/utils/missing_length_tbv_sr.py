"""Missing vessel length and blood volume of the super-resolution graphs, as % of the original.

Same graphs, matching and tolerance as ``centerline_coverage_sr.py`` (the configuration is imported from there).
Four numbers per scan and factor, each as a percentage of the original's total:

* Missed centerline: reference centerline length that no predicted centerline covers within the tolerance
  (= 100 * (1 - recall)).
* Missed blood volume: the same uncovered reference samples, weighted by vessel cross-section
  pi * r^2 (r = the edge's ``radius_avg``). This is the missed part of TBV, because VesselVio's TBV is
  sum(pi * radius_avg^2 * length) over all edges (``feature_processing``, edge attribute ``volume``).
* Length deficit: (L_orig - L_SR) / L_orig, total graph length, no spatial matching.
* TBV deficit: (TBV_orig - TBV_SR) / TBV_orig, TBV = sum of edge ``volume``, no spatial matching.

Positive values mean the SR volume has less than the original. The deficits can be negative (the SR
graph has more length / volume) and can be small even when much is missed, if the SR adds vessels elsewhere.

Outputs in OUTPUT_DIR: ``missing_length_tbv.csv`` (per scan) and ``missing_length_tbv_summary.csv``
(mean and SD per factor, the table that is printed).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vvl.utils.centerline_coverage_erik import compare_graphs, RadiusTolerance
from vvl.utils.centerline_coverage_sr import load

# ---------------------------------------------------------------------------- configuration

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
TOLERANCE = RadiusTolerance(radius_factor=1.0, minimum=0.006, maximum=None)
OUTPUT_DIR = STUDY_DIR / "missing_length_tbv_vesnet"
# Graph folders, spacing, sampling and tolerance are taken from centerline_coverage_sr.py

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

METRICS = {
    "missed_length_pct": "Missed centerline length",
    "missed_tbv_pct": "Missed blood volume",
    "length_deficit_pct": "Length deficit",
    "tbv_deficit_pct": "TBV deficit",
}


def total_blood_volume(graph) -> float:
    return float(sum(data["volume"] for *_, data in graph.edges(data=True)))


def evaluate(reference, prediction) -> dict:
    result = compare_graphs(
        reference, prediction,
        reference_spacing_xyz=SPACING_XYZ, prediction_spacing_xyz=SPACING_XYZ,
        sample_step=SAMPLE_STEP, tolerance=TOLERANCE,
    )
    ref_edges = [e for e in result.reference_edges if e.role == "reference"]
    cross_section = np.array([np.pi * e.radius_mean**2 for e in ref_edges])
    missed_volume = float(np.sum(cross_section * [e.uncovered_length for e in ref_edges]))
    sampled_volume = float(np.sum(cross_section * [e.total_length for e in ref_edges]))

    tbv_ref, tbv_pred = total_blood_volume(reference), total_blood_volume(prediction)
    return {
        "missed_length_pct": 100 * result.reference.uncovered_length / result.reference_length,
        "missed_tbv_pct": 100 * missed_volume / sampled_volume,
        "length_deficit_pct": 100 * (result.reference_length - result.prediction_length) / result.reference_length,
        "tbv_deficit_pct": 100 * (tbv_ref - tbv_pred) / tbv_ref,
        "reference_length_mm": result.reference_length,
        "prediction_length_mm": result.prediction_length,
        "missed_length_mm": result.reference.uncovered_length,
        "reference_tbv_mm3": tbv_ref,
        "prediction_tbv_mm3": tbv_pred,
        "missed_tbv_mm3": missed_volume,
        "reference_tbv_sampled_mm3": sampled_volume,  # sum(pi r^2 L) over resampled edges, ~= reference_tbv_mm3
    }


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = list(PREDICTION_DIRS)
    stats = df.groupby("label")[list(METRICS)].agg(["mean", "std"])
    table = pd.DataFrame(
        {lbl: [f"{stats.loc[lbl, (m, 'mean')]:.1f} ± {stats.loc[lbl, (m, 'std')]:.1f}" for m in METRICS]
         for lbl in labels if lbl in stats.index},
        index=list(METRICS.values()),
    )
    table.index.name = "% of original"
    return table


def main():
    reference_paths = sorted(REFERENCE_DIR.glob("*.pkl"))
    if not reference_paths:
        raise FileNotFoundError(f"No reference graphs in {REFERENCE_DIR}")
    print(f"{len(reference_paths)} reference graphs, factors {list(PREDICTION_DIRS)}")

    rows = []
    for i, ref_path in enumerate(reference_paths, start=1):
        reference = load(ref_path)
        for label, pred_dir in PREDICTION_DIRS.items():
            pred_path = pred_dir / ref_path.name
            if not pred_path.is_file():
                print(f"  missing {label} graph for {ref_path.stem}, skipping")
                continue
            rows.append({"sample": ref_path.stem, "label": label, **evaluate(reference, load(pred_path))})
        print(f"[{i}/{len(reference_paths)}] {ref_path.stem[:60]}", end="\r")

    df = pd.DataFrame(rows)
    table = summary_table(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "missing_length_tbv.csv", index=False)
    table.to_csv(OUTPUT_DIR / "missing_length_tbv_summary.csv", encoding="utf-8-sig")  # BOM so Excel shows ±

    counts = df.groupby("label").size()
    print(f"\n\nMissing vs. original, mean ± SD over scans (n = {', '.join(f'{l}: {counts.get(l, 0)}' for l in PREDICTION_DIRS)})")
    print(f"Tolerance: {TOLERANCE}\n")
    print(table.to_string())
    print(f"\nResults -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
