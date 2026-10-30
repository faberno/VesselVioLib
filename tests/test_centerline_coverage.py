import networkx as nx
import pytest

from vvl.utils.centerline_coverage_erik import (
    RadiusTolerance,
    compare_graphs,
    print_label_summary,
    summarize_by_label,
)


def _add_edge(
    graph,
    name,
    start,
    end,
    *,
    radius,
):
    u = f"{name}_u"
    v = f"{name}_v"
    graph.add_node(u, X=start[0], Y=start[1], Z=start[2])
    graph.add_node(v, X=end[0], Y=end[1], Z=end[2])
    graph.add_edge(
        u,
        v,
        coords_list=[start, end],
        radius_avg=radius,
    )


def _compare(
    reference,
    prediction,
    *,
    tolerance,
    reference_spacing=(1.0, 1.0, 1.0),
    prediction_spacing=(1.0, 1.0, 1.0),
):
    return compare_graphs(
        reference,
        prediction,
        reference_spacing_xyz=reference_spacing,
        prediction_spacing_xyz=prediction_spacing,
        sample_step=0.25,
        tolerance=tolerance,
    )


def test_identical_centerlines_have_complete_coverage():
    reference = nx.Graph()
    prediction = nx.Graph()
    _add_edge(reference, "ref", (0, 0, 0), (10, 0, 0), radius=0.5)
    _add_edge(prediction, "pred", (0, 0, 0), (10, 0, 0), radius=0.5)

    result = _compare(
        reference,
        prediction,
        tolerance=RadiusTolerance(radius_factor=1.0, minimum=0.0, maximum=None),
    )

    assert result.reference.fraction == pytest.approx(1.0)
    assert result.prediction.fraction == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.length_bias_percent == pytest.approx(0.0)


def test_reference_radius_controls_displacement_tolerance():
    thin_reference = nx.Graph()
    thick_reference = nx.Graph()
    prediction = nx.Graph()
    _add_edge(thin_reference, "thin", (0, 0, 0), (10, 0, 0), radius=0.5)
    _add_edge(thick_reference, "thick", (0, 0, 0), (10, 0, 0), radius=2.0)
    _add_edge(prediction, "pred", (0, 1, 0), (10, 1, 0), radius=100.0)
    tolerance = RadiusTolerance(radius_factor=1.0, minimum=0.0, maximum=None)

    thin_result = _compare(thin_reference, prediction, tolerance=tolerance)
    thick_result = _compare(thick_reference, prediction, tolerance=tolerance)

    assert thin_result.reference.fraction == pytest.approx(0.0)
    assert thin_result.prediction.fraction == pytest.approx(0.0)
    assert thick_result.reference.fraction == pytest.approx(1.0)
    assert thick_result.prediction.fraction == pytest.approx(1.0)


def test_missing_reference_vessel_reduces_recall_not_precision():
    reference = nx.Graph()
    prediction = nx.Graph()
    _add_edge(reference, "shared_ref", (0, 0, 0), (10, 0, 0), radius=0.5)
    _add_edge(reference, "missing_ref", (0, 10, 0), (10, 10, 0), radius=0.5)
    _add_edge(prediction, "shared_pred", (0, 0, 0), (10, 0, 0), radius=0.5)

    result = _compare(
        reference,
        prediction,
        tolerance=RadiusTolerance(radius_factor=1.0, minimum=0.0, maximum=None),
    )

    assert result.reference.fraction == pytest.approx(0.5)
    assert result.prediction.fraction == pytest.approx(1.0)
    assert result.reference.uncovered_length == pytest.approx(10.0)
    assert result.prediction.uncovered_length == pytest.approx(0.0)
    assert result.length_bias_percent == pytest.approx(-50.0)


def test_excess_prediction_vessel_reduces_precision_not_recall():
    reference = nx.Graph()
    prediction = nx.Graph()
    _add_edge(reference, "shared_ref", (0, 0, 0), (10, 0, 0), radius=0.5)
    _add_edge(prediction, "shared_pred", (0, 0, 0), (10, 0, 0), radius=0.5)
    _add_edge(prediction, "excess_pred", (0, 10, 0), (10, 10, 0), radius=0.5)

    result = _compare(
        reference,
        prediction,
        tolerance=RadiusTolerance(radius_factor=1.0, minimum=0.0, maximum=None),
    )

    assert result.reference.fraction == pytest.approx(1.0)
    assert result.prediction.fraction == pytest.approx(0.5)
    assert result.reference.uncovered_length == pytest.approx(0.0)
    assert result.prediction.uncovered_length == pytest.approx(10.0)
    assert result.length_bias_percent == pytest.approx(100.0)


def test_each_graph_spacing_is_applied_before_matching():
    reference = nx.Graph()
    prediction = nx.Graph()
    _add_edge(reference, "ref", (0, 0, 0), (10, 0, 0), radius=0.3)
    # Two prediction voxels in Y equal 0.2 physical units.
    _add_edge(prediction, "pred", (0, 2, 0), (10, 2, 0), radius=0.3)

    result = _compare(
        reference,
        prediction,
        tolerance=RadiusTolerance(radius_factor=1.0, minimum=0.0, maximum=None),
        reference_spacing=(0.1, 0.1, 0.1),
        prediction_spacing=(0.1, 0.1, 0.1),
    )

    assert result.reference.fraction == pytest.approx(1.0)
    assert result.prediction.fraction == pytest.approx(1.0)
    assert result.reference.mean_distance == pytest.approx(0.2)


def test_edge_subdivision_does_not_change_total_length_or_coverage():
    reference = nx.Graph()
    prediction = nx.Graph()
    _add_edge(reference, "ref", (0, 0, 0), (10, 0, 0), radius=0.5)
    _add_edge(prediction, "left", (0, 0, 0), (4, 0, 0), radius=0.5)
    _add_edge(prediction, "right", (4, 0, 0), (10, 0, 0), radius=0.5)

    result = _compare(
        reference,
        prediction,
        tolerance=RadiusTolerance(radius_factor=1.0, minimum=0.0, maximum=None),
    )

    assert result.reference.fraction == pytest.approx(1.0)
    assert result.prediction.fraction == pytest.approx(1.0)
    assert result.reference_length == pytest.approx(10.0)
    assert result.prediction_length == pytest.approx(10.0)


def test_label_summary_pools_lengths_in_configured_order(capsys):
    rows = [
        {
            "label": "i9",
            "reference_length": 100.0,
            "prediction_length": 80.0,
            "missed_reference_length": 30.0,
            "excess_prediction_length": 10.0,
        },
        {
            "label": "i9",
            "reference_length": 300.0,
            "prediction_length": 320.0,
            "missed_reference_length": 50.0,
            "excess_prediction_length": 20.0,
        },
        {
            "label": "i16",
            "reference_length": 200.0,
            "prediction_length": 220.0,
            "missed_reference_length": 20.0,
            "excess_prediction_length": 40.0,
        },
    ]

    summary = summarize_by_label(rows, ["i9", "i16", "i25"])

    assert [row["label"] for row in summary] == ["i9", "i16", "i25"]
    assert summary[0]["n_samples"] == 2
    assert summary[0]["recall_percent"] == pytest.approx(80.0)
    assert summary[0]["precision_percent"] == pytest.approx(92.5)
    assert summary[0]["length_bias_percent"] == pytest.approx(0.0)
    assert summary[2]["n_samples"] == 0

    print_label_summary(rows, ["i9", "i16", "i25"])
    output = capsys.readouterr().out
    assert "Pooled centerline coverage by prediction tier" in output
    assert "i9" in output
    assert "i16" in output
    assert "i25" in output
