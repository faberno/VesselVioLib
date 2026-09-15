"""
Radius-adaptive, length-weighted coverage of vascular centerline graphs.

This module deliberately compares centerline geometry rather than graph edge
IDs.  It is therefore insensitive to a vessel being represented by one edge
in one graph and several degree-2 edges in the other graph.

The graph coordinates written by VesselVio are voxel-index XYZ coordinates,
whereas radii are stored in physical units.  ``spacing_xyz`` must therefore
convert coordinates into the same physical unit as ``radius_avg`` and
``radii_list`` (millimetres for the configuration at the bottom of this file).

The executable section contains normal Python configuration rather than an
argument parser so it can be run and edited directly in VS Code.
"""

from __future__ import annotations

import ast
import csv
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RadiusTolerance:
    """
    Convert a local reference-vessel radius into a matching tolerance.

    ``tolerance = clip(radius_factor * reference_radius, minimum, maximum)``

    The reference radius is used in both comparison directions.  In
    particular, a prediction with an inflated radius cannot increase its own
    matching tolerance.
    """

    radius_factor: float = 1.0
    minimum: float = 0.006
    maximum: Optional[float] = 0.060

    def __post_init__(self) -> None:
        if self.radius_factor < 0:
            raise ValueError("radius_factor must be non-negative")
        if self.minimum < 0:
            raise ValueError("minimum must be non-negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum")

    def evaluate(self, reference_radii: np.ndarray) -> np.ndarray:
        """Return tolerances in the same physical unit as the input radii."""
        radii = np.asarray(reference_radii, dtype=float)
        if np.any(~np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError("Reference radii must be finite and non-negative")

        tolerances = np.maximum(self.radius_factor * radii, self.minimum)
        if self.maximum is not None:
            tolerances = np.minimum(tolerances, self.maximum)
        return tolerances


@dataclass(frozen=True)
class SampledCenterlines:
    """Uniform arc-length samples representing all graph edges."""

    points: np.ndarray
    radii: np.ndarray
    weights: np.ndarray
    edge_ids: np.ndarray

    @property
    def total_length(self) -> float:
        return float(np.sum(self.weights))


@dataclass(frozen=True)
class DirectionalCoverage:
    """Coverage of one graph by the other graph."""

    fraction: float
    covered_length: float
    uncovered_length: float
    total_length: float
    mean_distance: float
    median_distance: float
    distance_p95: float
    n_samples: int


@dataclass(frozen=True)
class EdgeCoverage:
    """Length-weighted coverage information for one source edge."""

    role: str
    edge_id: str
    fraction: float
    covered_length: float
    uncovered_length: float
    total_length: float
    mean_distance: float
    radius_mean: float


@dataclass(frozen=True)
class CoverageResult:
    """Complete bidirectional centerline comparison."""

    reference: DirectionalCoverage
    prediction: DirectionalCoverage
    reference_length: float
    prediction_length: float
    length_ratio: float
    length_bias: float
    length_bias_percent: float
    f1: float
    reference_edges: Tuple[EdgeCoverage, ...]
    prediction_edges: Tuple[EdgeCoverage, ...]

    def summary_dict(self) -> Dict[str, float]:
        """Return a flat dictionary suitable for a CSV row."""
        return {
            "reference_length": self.reference_length,
            "prediction_length": self.prediction_length,
            "length_ratio": self.length_ratio,
            "length_bias": self.length_bias,
            "length_bias_percent": self.length_bias_percent,
            "reference_recall": self.reference.fraction,
            "prediction_precision": self.prediction.fraction,
            "centerline_f1": self.f1,
            "missed_reference_length": self.reference.uncovered_length,
            "excess_prediction_length": self.prediction.uncovered_length,
            "reference_mean_distance": self.reference.mean_distance,
            "reference_median_distance": self.reference.median_distance,
            "reference_distance_p95": self.reference.distance_p95,
            "prediction_mean_distance": self.prediction.mean_distance,
            "prediction_median_distance": self.prediction.median_distance,
            "prediction_distance_p95": self.prediction.distance_p95,
        }


def load_graph(path: Path) -> nx.Graph:
    """
    Load a VesselVio pickle or a NetworkX GraphML file.

    Pickle files are trusted input only.  Python pickle is unsafe for files
    obtained from untrusted sources.
    """
    path = Path(path)
    if path.suffix.lower() == ".graphml":
        return nx.read_graphml(path)

    with path.open("rb") as file:
        graph = pickle.load(file)

    if isinstance(graph, nx.Graph):
        return graph
    if hasattr(graph, "to_networkx"):
        return graph.to_networkx()
    raise TypeError(f"Unsupported graph object in {path}: {type(graph)!r}")


def _parse_numeric_sequence(value: Any, attribute: str) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    result = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{attribute} contains non-finite values")
    return result


def _node_xyz(graph: nx.Graph, node: Any) -> np.ndarray:
    data = graph.nodes[node]
    try:
        point = np.array(
            [float(data["X"]), float(data["Y"]), float(data["Z"])],
            dtype=float,
        )
    except KeyError as error:
        raise KeyError(f"Node {node!r} is missing coordinate {error.args[0]!r}") from error
    if np.any(~np.isfinite(point)):
        raise ValueError(f"Node {node!r} contains non-finite coordinates")
    return point


def _iter_edges(
    graph: nx.Graph,
) -> Iterable[Tuple[Any, Any, Optional[Any], Mapping[str, Any]]]:
    if graph.is_multigraph():
        for u, v, key, data in graph.edges(keys=True, data=True):
            yield u, v, key, data
    else:
        for u, v, data in graph.edges(data=True):
            yield u, v, None, data


def _edge_id(u: Any, v: Any, key: Optional[Any]) -> str:
    if key is None:
        return f"{u!r}--{v!r}"
    return f"{u!r}--{v!r}--key={key!r}"


def _edge_polyline_xyz(
    graph: nx.Graph,
    u: Any,
    v: Any,
    data: Mapping[str, Any],
    spacing_xyz: np.ndarray,
) -> np.ndarray:
    if "coords_list" in data:
        points = _parse_numeric_sequence(data["coords_list"], "coords_list")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Edge {(u, v)!r} coords_list must have shape (N, 3), "
                f"got {points.shape}"
            )
    else:
        points = np.vstack((_node_xyz(graph, u), _node_xyz(graph, v)))

    if len(points) < 2:
        raise ValueError(f"Edge {(u, v)!r} has fewer than two centerline points")
    return points * spacing_xyz


def _edge_radius_profile(
    data: Mapping[str, Any],
    n_control_points: int,
    radius_scale: float,
) -> np.ndarray:
    if "radii_list" in data:
        radii = _parse_numeric_sequence(data["radii_list"], "radii_list").reshape(-1)
        if len(radii) == 0:
            raise ValueError("radii_list must not be empty")
    elif "radius_avg" in data:
        radii = np.full(n_control_points, float(data["radius_avg"]), dtype=float)
    else:
        raise KeyError("Edge is missing both radii_list and radius_avg")

    radii = radii * radius_scale
    if np.any(radii < 0):
        raise ValueError("Vessel radii must be non-negative")
    return radii


def _resample_edge(
    points: np.ndarray,
    radii: np.ndarray,
    sample_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample interval midpoints and exact arc-length weights along one polyline.

    The weights sum exactly to the polyline length, preventing edge endpoints
    or densely sampled source polylines from receiving disproportionate weight.
    """
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep_segments = segment_lengths > 0
    if not np.any(keep_segments):
        return (
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    # Remove repeated consecutive points while retaining the final point.
    keep_points = np.concatenate(([True], keep_segments))
    points = points[keep_points]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])

    n_samples = max(1, int(np.ceil(total_length / sample_step)))
    weight = total_length / n_samples
    distances = (np.arange(n_samples, dtype=float) + 0.5) * weight

    segment_indices = np.searchsorted(cumulative, distances, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(segment_lengths) - 1)
    local_fraction = (
        (distances - cumulative[segment_indices])
        / segment_lengths[segment_indices]
    )
    sampled_points = (
        points[segment_indices]
        + local_fraction[:, None]
        * (points[segment_indices + 1] - points[segment_indices])
    )

    # radii_list can originate from a differently sampled centerline.  Map it
    # over normalized arc position; radius_avg naturally remains constant.
    radius_positions = np.linspace(0.0, total_length, len(radii))
    sampled_radii = np.interp(distances, radius_positions, radii)
    weights = np.full(n_samples, weight, dtype=float)
    return sampled_points, sampled_radii, weights


def sample_graph_centerlines(
    graph: nx.Graph,
    *,
    spacing_xyz: Sequence[float],
    sample_step: float,
    radius_scale: float = 1.0,
) -> SampledCenterlines:
    """
    Resample all graph edges in a shared physical coordinate system.

    Parameters
    ----------
    spacing_xyz:
        Physical size of one graph-coordinate unit in X, Y, Z order.
    sample_step:
        Maximum distance between interval-midpoint samples, in physical units.
    radius_scale:
        Optional conversion applied to stored radii.  Usually 1.0 because
        VesselVio already stores radii in physical units.
    """
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three finite positive values")
    if not np.isfinite(sample_step) or sample_step <= 0:
        raise ValueError("sample_step must be finite and positive")
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError("radius_scale must be finite and positive")

    all_points: List[np.ndarray] = []
    all_radii: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []
    all_edge_ids: List[np.ndarray] = []

    for u, v, key, data in _iter_edges(graph):
        points = _edge_polyline_xyz(graph, u, v, data, spacing)
        radii = _edge_radius_profile(data, len(points), radius_scale)
        sampled_points, sampled_radii, weights = _resample_edge(
            points,
            radii,
            sample_step,
        )
        if len(sampled_points) == 0:
            continue

        edge_id = _edge_id(u, v, key)
        all_points.append(sampled_points)
        all_radii.append(sampled_radii)
        all_weights.append(weights)
        all_edge_ids.append(np.full(len(weights), edge_id, dtype=object))

    if not all_points:
        raise ValueError("Graph has no non-zero-length edges to compare")

    return SampledCenterlines(
        points=np.vstack(all_points),
        radii=np.concatenate(all_radii),
        weights=np.concatenate(all_weights),
        edge_ids=np.concatenate(all_edge_ids),
    )


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _summarize_direction(
    weights: np.ndarray,
    distances: np.ndarray,
    covered: np.ndarray,
) -> DirectionalCoverage:
    total = float(np.sum(weights))
    covered_length = float(np.sum(weights[covered]))
    return DirectionalCoverage(
        fraction=covered_length / total,
        covered_length=covered_length,
        uncovered_length=total - covered_length,
        total_length=total,
        mean_distance=float(np.average(distances, weights=weights)),
        median_distance=_weighted_quantile(distances, weights, 0.50),
        distance_p95=_weighted_quantile(distances, weights, 0.95),
        n_samples=len(weights),
    )


def _summarize_edges(
    role: str,
    samples: SampledCenterlines,
    distances: np.ndarray,
    covered: np.ndarray,
) -> Tuple[EdgeCoverage, ...]:
    rows: List[EdgeCoverage] = []
    for edge_id in dict.fromkeys(samples.edge_ids.tolist()):
        mask = samples.edge_ids == edge_id
        weights = samples.weights[mask]
        edge_covered = covered[mask]
        total = float(np.sum(weights))
        covered_length = float(np.sum(weights[edge_covered]))
        rows.append(
            EdgeCoverage(
                role=role,
                edge_id=str(edge_id),
                fraction=covered_length / total,
                covered_length=covered_length,
                uncovered_length=total - covered_length,
                total_length=total,
                mean_distance=float(np.average(distances[mask], weights=weights)),
                radius_mean=float(np.average(samples.radii[mask], weights=weights)),
            )
        )
    return tuple(rows)


def compare_sampled_centerlines(
    reference: SampledCenterlines,
    prediction: SampledCenterlines,
    tolerance: RadiusTolerance,
) -> CoverageResult:
    """
    Calculate length-weighted reference recall and prediction precision.

    For reference recall, tolerance comes from each reference sample.  For
    prediction precision, each prediction sample uses the radius of its nearest
    reference sample.  The same reference-radius rule therefore governs both
    directions.
    """
    prediction_tree = cKDTree(prediction.points)
    reference_distances, _ = prediction_tree.query(reference.points, k=1)
    reference_tolerances = tolerance.evaluate(reference.radii)
    reference_covered = reference_distances <= reference_tolerances

    reference_tree = cKDTree(reference.points)
    prediction_distances, nearest_reference = reference_tree.query(
        prediction.points,
        k=1,
    )
    prediction_tolerances = tolerance.evaluate(
        reference.radii[np.asarray(nearest_reference, dtype=int)]
    )
    prediction_covered = prediction_distances <= prediction_tolerances

    reference_summary = _summarize_direction(
        reference.weights,
        np.asarray(reference_distances),
        reference_covered,
    )
    prediction_summary = _summarize_direction(
        prediction.weights,
        np.asarray(prediction_distances),
        prediction_covered,
    )

    reference_length = reference.total_length
    prediction_length = prediction.total_length
    length_bias = prediction_length - reference_length
    length_ratio = prediction_length / reference_length
    recall = reference_summary.fraction
    precision = prediction_summary.fraction
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return CoverageResult(
        reference=reference_summary,
        prediction=prediction_summary,
        reference_length=reference_length,
        prediction_length=prediction_length,
        length_ratio=length_ratio,
        length_bias=length_bias,
        length_bias_percent=100.0 * length_bias / reference_length,
        f1=f1,
        reference_edges=_summarize_edges(
            "reference",
            reference,
            np.asarray(reference_distances),
            reference_covered,
        ),
        prediction_edges=_summarize_edges(
            "prediction",
            prediction,
            np.asarray(prediction_distances),
            prediction_covered,
        ),
    )


def compare_graphs(
    reference_graph: nx.Graph,
    prediction_graph: nx.Graph,
    *,
    reference_spacing_xyz: Sequence[float],
    prediction_spacing_xyz: Sequence[float],
    sample_step: float,
    tolerance: RadiusTolerance,
    reference_radius_scale: float = 1.0,
    prediction_radius_scale: float = 1.0,
) -> CoverageResult:
    """Resample and compare two vascular graphs."""
    reference = sample_graph_centerlines(
        reference_graph,
        spacing_xyz=reference_spacing_xyz,
        sample_step=sample_step,
        radius_scale=reference_radius_scale,
    )
    prediction = sample_graph_centerlines(
        prediction_graph,
        spacing_xyz=prediction_spacing_xyz,
        sample_step=sample_step,
        radius_scale=prediction_radius_scale,
    )
    return compare_sampled_centerlines(reference, prediction, tolerance)


def compare_graph_files(
    reference_path: Path,
    prediction_path: Path,
    **kwargs: Any,
) -> CoverageResult:
    """Load and compare two graph files."""
    return compare_graphs(
        load_graph(reference_path),
        load_graph(prediction_path),
        **kwargs,
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_label(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Pool sample-level lengths into one summary row per prediction label.

    Pooling lengths is preferable to averaging sample percentages: every
    physical unit of centerline receives equal weight regardless of which
    sample it belongs to.
    """
    summaries: List[Dict[str, Any]] = []
    for label in labels:
        label_rows = [row for row in rows if row["label"] == label]
        if not label_rows:
            summaries.append(
                {
                    "label": label,
                    "n_samples": 0,
                    "recall_percent": float("nan"),
                    "precision_percent": float("nan"),
                    "f1_percent": float("nan"),
                    "length_bias_percent": float("nan"),
                    "missed_length": float("nan"),
                    "excess_length": float("nan"),
                }
            )
            continue

        reference_length = sum(float(row["reference_length"]) for row in label_rows)
        prediction_length = sum(float(row["prediction_length"]) for row in label_rows)
        missed_length = sum(
            float(row["missed_reference_length"]) for row in label_rows
        )
        excess_length = sum(
            float(row["excess_prediction_length"]) for row in label_rows
        )

        recall = (
            (reference_length - missed_length) / reference_length
            if reference_length > 0
            else float("nan")
        )
        precision = (
            (prediction_length - excess_length) / prediction_length
            if prediction_length > 0
            else float("nan")
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        length_bias_percent = (
            100.0 * (prediction_length - reference_length) / reference_length
            if reference_length > 0
            else float("nan")
        )

        summaries.append(
            {
                "label": label,
                "n_samples": len(label_rows),
                "recall_percent": 100.0 * recall,
                "precision_percent": 100.0 * precision,
                "f1_percent": 100.0 * f1,
                "length_bias_percent": length_bias_percent,
                "missed_length": missed_length,
                "excess_length": excess_length,
            }
        )
    return summaries


def print_label_summary(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> None:
    """Print a compact pooled comparison table for all configured tiers."""
    summaries = summarize_by_label(rows, labels)
    if not any(row["n_samples"] > 0 for row in summaries):
        print("\nNo completed comparisons to summarize.")
        return

    header = (
        f"{'tier':<8}"
        f"{'samples':>9}"
        f"{'recall %':>12}"
        f"{'precision %':>14}"
        f"{'F1 %':>10}"
        f"{'length bias %':>16}"
        f"{'missed length':>16}"
        f"{'excess length':>16}"
    )
    print("\nPooled centerline coverage by prediction tier")
    print(
        "Lengths are pooled across samples; length quantities use the configured "
        "physical unit."
    )
    print(header)
    print("-" * len(header))
    for row in summaries:
        if row["n_samples"] == 0:
            print(f"{row['label']:<8}{0:>9d}{'--':>12}{'--':>14}{'--':>10}"
                  f"{'--':>16}{'--':>16}{'--':>16}")
            continue
        print(
            f"{row['label']:<8}"
            f"{row['n_samples']:>9d}"
            f"{row['recall_percent']:>12.2f}"
            f"{row['precision_percent']:>14.2f}"
            f"{row['f1_percent']:>10.2f}"
            f"{row['length_bias_percent']:>+16.2f}"
            f"{row['missed_length']:>16.4f}"
            f"{row['excess_length']:>16.4f}"
        )


def run_batch(
    *,
    reference_dir: Path,
    prediction_dirs: Mapping[str, Path],
    output_dir: Path,
    reference_spacing_xyz: Sequence[float],
    prediction_spacing_xyz: Mapping[str, Sequence[float]],
    sample_step: float,
    tolerance: RadiusTolerance,
    file_pattern: str = "*.pkl",
) -> List[Dict[str, Any]]:
    """Compare every reference graph with same-named prediction graphs."""
    reference_paths = sorted(Path(reference_dir).glob(file_pattern))
    if not reference_paths:
        raise FileNotFoundError(
            f"No reference graphs matching {file_pattern!r} in {reference_dir}"
        )

    summary_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []

    for reference_path in reference_paths:
        for label, prediction_dir in prediction_dirs.items():
            prediction_path = Path(prediction_dir) / reference_path.name
            if not prediction_path.exists():
                print(f"Skipping missing prediction: {prediction_path}")
                continue
            if label not in prediction_spacing_xyz:
                raise KeyError(f"No prediction spacing configured for {label!r}")

            print(f"Comparing {reference_path.name}: reference vs {label}")
            result = compare_graph_files(
                reference_path,
                prediction_path,
                reference_spacing_xyz=reference_spacing_xyz,
                prediction_spacing_xyz=prediction_spacing_xyz[label],
                sample_step=sample_step,
                tolerance=tolerance,
            )
            summary_rows.append(
                {
                    "sample": reference_path.name,
                    "label": label,
                    **result.summary_dict(),
                }
            )
            for edge in (*result.reference_edges, *result.prediction_edges):
                edge_rows.append(
                    {
                        "sample": reference_path.name,
                        "label": label,
                        **asdict(edge),
                    }
                )

            print(
                f"  recall={result.reference.fraction:.3f}, "
                f"precision={result.prediction.fraction:.3f}, "
                f"F1={result.f1:.3f}, "
                f"length bias={result.length_bias_percent:+.1f}%"
            )

    output_dir = Path(output_dir)
    _write_csv(output_dir / "centerline_coverage_summary.csv", summary_rows)
    _write_csv(output_dir / "centerline_coverage_edges.csv", edge_rows)
    print_label_summary(summary_rows, list(prediction_dirs))
    return summary_rows


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

BASE = Path(r"E:\sr_data\532\cuff_analysis\graph_comparison_cuff_base")

REFERENCE_DIR = BASE / "original" / "vesselvio" / "Graphs"
PREDICTION_DIRS = {
    "i9": BASE / "i9" / "vesselvio",
    "i16": BASE / "i16" / "vesselvio",
    "i25": BASE / "i25" / "vesselvio" / "Graphs",
}
OUTPUT_DIR = BASE / "centerline_coverage"

# IMPORTANT: VesselVio graph coordinates are XYZ, while NumPy volume spacing is
# often supplied as ZYX.  The values below correspond to a ZYX image spacing
# of (0.003, 0.012, 0.012) mm.  Change each tier independently if its graph is
# expressed on a different voxel grid.
REFERENCE_SPACING_XYZ = (0.012, 0.012, 0.003)
PREDICTION_SPACING_XYZ = {
    "i9": (0.012, 0.012, 0.003),
    "i16": (0.012, 0.012, 0.003),
    "i25": (0.012, 0.012, 0.003),
}

# Samples are interval midpoints.  Their weights exactly recover geometric
# centerline length.  A 0.003 mm step matches the finest configured voxel axis.
SAMPLE_STEP = 0.003

# Default tolerance: one local reference radius, with a 6 µm localization
# floor and a 60 µm cap.  A threshold sweep should be used for final reporting.
TOLERANCE = RadiusTolerance(
    radius_factor=.5,
    minimum=0.006,
    maximum=None,
)


if __name__ == "__main__":
    run_batch(
        reference_dir=REFERENCE_DIR,
        prediction_dirs=PREDICTION_DIRS,
        output_dir=OUTPUT_DIR,
        reference_spacing_xyz=REFERENCE_SPACING_XYZ,
        prediction_spacing_xyz=PREDICTION_SPACING_XYZ,
        sample_step=SAMPLE_STEP,
        tolerance=TOLERANCE,
    )
