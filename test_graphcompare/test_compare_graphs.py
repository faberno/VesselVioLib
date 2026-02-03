"""
Test suite for vvl.compare_graphs using synthetic graph pairs.

Each scenario builds a controlled pair of graphs simulating a specific
segmentation difference, then asserts the comparison produces correct
matching, splits/merges, and surplus classification.
"""

import networkx as nx
import numpy as np
import pytest

from vvl.compare_graphs import (
    compare_graphs,
    GraphComparisonResult,
    pair_nodes_by_proximity,
    pair_edges_by_nodes,
    find_split_merge_matches,
    compute_edge_differences,
    build_difference_graph,
    build_surplus_graph,
    get_largest_differences,
    get_missing_edges_info,
    sample_edge_path,
    compute_edge_overlap,
    edge_in_set,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EDGE_FEATURES = [
    "length", "radius_avg", "radius_max", "radius_min",
    "tortuosity", "volume", "surface_area",
]


def _make_node(graph, node_id, x, y, z):
    """Add a node with XYZ coordinates."""
    graph.add_node(node_id, X=float(x), Y=float(y), Z=float(z))


def _make_edge(graph, u, v, **overrides):
    """Add an edge with default vessel features, optionally overridden."""
    p1 = np.array([float(graph.nodes[u]["X"]),
                    float(graph.nodes[u]["Y"]),
                    float(graph.nodes[u]["Z"])])
    p2 = np.array([float(graph.nodes[v]["X"]),
                    float(graph.nodes[v]["Y"]),
                    float(graph.nodes[v]["Z"])])
    length = float(np.linalg.norm(p2 - p1))
    defaults = {
        "length": length,
        "radius_avg": 0.05,
        "radius_max": 0.07,
        "radius_min": 0.03,
        "tortuosity": 1.0,
        "volume": 0.001,
        "surface_area": 0.01,
    }
    defaults.update(overrides)
    graph.add_edge(u, v, **defaults)


def _make_line(graph, prefix, coords, **edge_kw):
    """
    Build a straight chain of nodes/edges.

    coords: list of (x,y,z) tuples.
    Returns list of node IDs created.
    """
    ids = []
    for i, (x, y, z) in enumerate(coords):
        nid = f"{prefix}_{i}"
        _make_node(graph, nid, x, y, z)
        ids.append(nid)
    for i in range(len(ids) - 1):
        _make_edge(graph, ids[i], ids[i + 1], **edge_kw)
    return ids


def _make_y_shape(graph, prefix, root, left, right, branch_pt=None):
    """
    Build a Y-shaped subgraph (bifurcation).

    root, left, right: (x,y,z) tuples for endpoints.
    branch_pt: (x,y,z) for the bifurcation; defaults to midpoint of root-left.
    """
    if branch_pt is None:
        branch_pt = tuple((np.array(root) + np.array(left)) / 2)

    ids = {}
    _make_node(graph, f"{prefix}_root", *root)
    _make_node(graph, f"{prefix}_bp", *branch_pt)
    _make_node(graph, f"{prefix}_left", *left)
    _make_node(graph, f"{prefix}_right", *right)

    _make_edge(graph, f"{prefix}_root", f"{prefix}_bp")
    _make_edge(graph, f"{prefix}_bp", f"{prefix}_left")
    _make_edge(graph, f"{prefix}_bp", f"{prefix}_right")

    return [f"{prefix}_root", f"{prefix}_bp",
            f"{prefix}_left", f"{prefix}_right"]


def _count_splits(result):
    return sum(1 for s in result.edge_splits if s.match_type == "split")


def _count_merges(result):
    return sum(1 for s in result.edge_splits if s.match_type == "merge")


# ---------------------------------------------------------------------------
# 1. Identical graphs  (perfect match, zero surplus)
# ---------------------------------------------------------------------------

class TestIdenticalGraphs:
    def _make_pair(self):
        g = nx.Graph()
        _make_line(g, "n", [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0)])
        return g, g.copy()

    def test_all_nodes_paired(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2)
        assert len(r.node_pairs) == g1.number_of_nodes()

    def test_all_edges_paired(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2)
        assert len(r.edge_pairs) == g1.number_of_edges()

    def test_no_unmatched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2)
        assert len(r.unmatched_nodes_g1) == 0
        assert len(r.unmatched_nodes_g2) == 0
        assert len(r.unmatched_edges_g1) == 0
        assert len(r.unmatched_edges_g2) == 0

    def test_no_splits(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2)
        assert len(r.edge_splits) == 0

    def test_surplus_empty(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2)
        assert r.surplus_graph.number_of_nodes() == 0
        assert r.surplus_graph.number_of_edges() == 0

    def test_zero_feature_diffs(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2)
        for _, _, attr in r.difference_graph.edges(data=True):
            for feat in EDGE_FEATURES:
                if f"{feat}_diff" in attr:
                    assert attr[f"{feat}_diff"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Uniform node jitter  (same topology, slightly displaced nodes)
# ---------------------------------------------------------------------------

class TestNodeJitter:
    def _make_pair(self, jitter=1.0):
        rng = np.random.default_rng(42)
        g1 = nx.Graph()
        g2 = nx.Graph()
        coords = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (10, 10, 0)]
        for i, (x, y, z) in enumerate(coords):
            _make_node(g1, f"n{i}", x, y, z)
            dx, dy, dz = rng.uniform(-jitter, jitter, 3)
            _make_node(g2, f"m{i}", x + dx, y + dy, z + dz)
        for u, v in [(0, 1), (1, 2), (1, 3)]:
            _make_edge(g1, f"n{u}", f"n{v}")
            _make_edge(g2, f"m{u}", f"m{v}")
        return g1, g2

    def test_all_matched_within_threshold(self):
        g1, g2 = self._make_pair(jitter=1.0)
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 4
        assert len(r.edge_pairs) == 3

    def test_none_matched_beyond_threshold(self):
        g1, g2 = self._make_pair(jitter=1.0)
        r = compare_graphs(g1, g2, max_node_distance=0.01)
        assert len(r.node_pairs) == 0


# ---------------------------------------------------------------------------
# 3. Edge split  (1 edge in g1 -> 2 edges in g2 via new intermediate node)
# ---------------------------------------------------------------------------

class TestEdgeSplit:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 20, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "M", 10, 0, 0)   # new intermediate
        _make_node(g2, "B2", 20, 0, 0)
        _make_edge(g2, "A2", "M")
        _make_edge(g2, "M", "B2")
        return g1, g2

    def test_split_detected(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert _count_splits(r) == 1

    def test_no_1to1_edge_pair(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.edge_pairs) == 0

    def test_endpoints_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert "A" in r.node_pairs
        assert "B" in r.node_pairs

    def test_intermediate_unmatched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert "M" in r.unmatched_nodes_g2


# ---------------------------------------------------------------------------
# 4. Edge merge  (2 edges in g1 -> 1 edge in g2, intermediate node removed)
# ---------------------------------------------------------------------------

class TestEdgeMerge:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "M", 10, 0, 0)
        _make_node(g1, "B", 20, 0, 0)
        _make_edge(g1, "A", "M")
        _make_edge(g1, "M", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "B2", 20, 0, 0)
        _make_edge(g2, "A2", "B2")
        return g1, g2

    def test_merge_detected(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert _count_merges(r) == 1

    def test_intermediate_unmatched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert "M" in r.unmatched_nodes_g1


# ---------------------------------------------------------------------------
# 5. Chain split  (1 edge -> 3 edges)
# ---------------------------------------------------------------------------

class TestChainSplit:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 30, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "P", 10, 0, 0)
        _make_node(g2, "Q", 20, 0, 0)
        _make_node(g2, "B2", 30, 0, 0)
        _make_edge(g2, "A2", "P")
        _make_edge(g2, "P", "Q")
        _make_edge(g2, "Q", "B2")
        return g1, g2

    def test_split_into_three(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0, max_split_hops=4)
        assert _count_splits(r) == 1
        split = [s for s in r.edge_splits if s.match_type == "split"][0]
        assert len(split.edges_g2) == 3


# ---------------------------------------------------------------------------
# 6. Added branch  (g2 has extra branch not in g1)
# ---------------------------------------------------------------------------

class TestAddedBranch:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_line(g1, "a", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])

        g2 = nx.Graph()
        _make_line(g2, "b", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])
        # extra branch off the middle node
        _make_node(g2, "extra", 10, 10, 0)
        _make_edge(g2, "b_1", "extra")
        return g1, g2

    def test_extra_branch_in_surplus(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert "extra" in r.unmatched_nodes_g2
        assert len(r.unmatched_edges_g2) >= 1

    def test_main_chain_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.edge_pairs) == 2


# ---------------------------------------------------------------------------
# 7. Removed branch  (g1 has branch not in g2)
# ---------------------------------------------------------------------------

class TestRemovedBranch:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_line(g1, "a", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])
        _make_node(g1, "spur", 10, 10, 0)
        _make_edge(g1, "a_1", "spur")

        g2 = nx.Graph()
        _make_line(g2, "b", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])
        return g1, g2

    def test_spur_in_surplus(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert "spur" in r.unmatched_nodes_g1
        assert len(r.unmatched_edges_g1) >= 1

    def test_main_chain_still_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.edge_pairs) == 2


# ---------------------------------------------------------------------------
# 8. Feature differences  (same topology, different edge attributes)
# ---------------------------------------------------------------------------

class TestFeatureDifferences:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 10, 0, 0)
        _make_edge(g1, "A", "B", radius_avg=0.05, volume=0.001)

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "B2", 10, 0, 0)
        _make_edge(g2, "A2", "B2", radius_avg=0.10, volume=0.004)
        return g1, g2

    def test_diff_computed(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        edges = list(r.difference_graph.edges(data=True))
        assert len(edges) == 1
        attr = edges[0][2]
        assert attr["radius_avg_diff"] == pytest.approx(0.05)
        assert attr["radius_avg_pct_diff"] == pytest.approx(100.0)

    def test_largest_differences(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        top = get_largest_differences(r, "radius_avg", top_n=5)
        assert len(top) == 1
        assert top[0]["diff"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 9. Zero-valued feature  (pct_diff should be inf)
# ---------------------------------------------------------------------------

class TestZeroFeature:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 10, 0, 0)
        _make_edge(g1, "A", "B", volume=0.0)

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "B2", 10, 0, 0)
        _make_edge(g2, "A2", "B2", volume=0.005)
        return g1, g2

    def test_pct_diff_is_inf(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        attr = list(r.difference_graph.edges(data=True))[0][2]
        assert np.isinf(attr["volume_pct_diff"])

    def test_inf_excluded_when_sorting_by_pct(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        # by_absolute=False -> sorts by pct_diff, inf is filtered
        top = get_largest_differences(r, "volume", top_n=5, by_absolute=False)
        assert len(top) == 0

    def test_abs_diff_still_returned(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        # by_absolute=True -> sorts by diff (0.005), not inf
        top = get_largest_differences(r, "volume", top_n=5, by_absolute=True)
        assert len(top) == 1


# ---------------------------------------------------------------------------
# 10. Empty graphs
# ---------------------------------------------------------------------------

class TestEmptyGraphs:
    def test_both_empty(self):
        g1, g2 = nx.Graph(), nx.Graph()
        r = compare_graphs(g1, g2)
        assert len(r.node_pairs) == 0
        assert r.surplus_graph.number_of_nodes() == 0

    def test_g1_empty(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        _make_node(g2, "A", 0, 0, 0)
        r = compare_graphs(g1, g2)
        assert len(r.unmatched_nodes_g2) == 1

    def test_g2_empty(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        r = compare_graphs(g1, g2)
        assert len(r.unmatched_nodes_g1) == 1


# ---------------------------------------------------------------------------
# 11. Single-node graphs
# ---------------------------------------------------------------------------

class TestSingleNode:
    def test_single_node_matched(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        _make_node(g1, "A", 5, 5, 5)
        _make_node(g2, "B", 5, 5, 5)
        r = compare_graphs(g1, g2)
        assert len(r.node_pairs) == 1

    def test_single_node_too_far(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g2, "B", 100, 100, 100)
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 0


# ---------------------------------------------------------------------------
# 12. Disconnected components  (one component matches, one does not)
# ---------------------------------------------------------------------------

class TestDisconnectedComponents:
    def _make_pair(self):
        g1 = nx.Graph()
        # Component 1: shared
        _make_line(g1, "s", [(0, 0, 0), (10, 0, 0)])
        # Component 2: only in g1
        _make_line(g1, "x", [(100, 100, 100), (110, 100, 100)])

        g2 = nx.Graph()
        # Component 1: shared
        _make_line(g2, "t", [(0, 0, 0), (10, 0, 0)])
        # Component 3: only in g2
        _make_line(g2, "y", [(200, 200, 200), (210, 200, 200)])
        return g1, g2

    def test_shared_component_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.edge_pairs) == 1  # shared component edge

    def test_unique_components_in_surplus(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.unmatched_edges_g1) == 1
        assert len(r.unmatched_edges_g2) == 1


# ---------------------------------------------------------------------------
# 13. Node at boundary of max_distance
# ---------------------------------------------------------------------------

class TestBoundaryDistance:
    def test_exactly_at_threshold_matched(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g2, "B", 5, 0, 0)
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 1

    def test_just_beyond_threshold_unmatched(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g2, "B", 5.01, 0, 0)
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 0


# ---------------------------------------------------------------------------
# 14. Competing matches  (two g1 nodes close to same g2 node)
# ---------------------------------------------------------------------------

class TestCompetingMatches:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 1, 0, 0)   # very close to A

        g2 = nx.Graph()
        _make_node(g2, "X", 0.5, 0, 0)  # equidistant to both
        return g1, g2

    def test_only_one_paired(self):
        """Hungarian should pair exactly one g1 node to X, leave other unmatched."""
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 1
        assert len(r.unmatched_nodes_g1) == 1

    def test_no_double_assignment(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assigned_g2 = list(r.node_pairs.values())
        assert len(assigned_g2) == len(set(assigned_g2))  # all unique


# ---------------------------------------------------------------------------
# 15. Bifurcation point shift  (Y-shape with shifted branch point)
# ---------------------------------------------------------------------------

class TestBifurcationShift:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_y_shape(g1, "g1",
                       root=(0, 0, 0), left=(20, 10, 0), right=(20, -10, 0),
                       branch_pt=(10, 0, 0))

        g2 = nx.Graph()
        _make_y_shape(g2, "g2",
                       root=(0, 0, 0), left=(20, 10, 0), right=(20, -10, 0),
                       branch_pt=(12, 0, 0))  # shifted 2 units
        return g1, g2

    def test_all_nodes_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 4

    def test_all_edges_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.edge_pairs) == 3


# ---------------------------------------------------------------------------
# 16. Bifurcation appears / disappears  (straight line vs Y-shape)
# ---------------------------------------------------------------------------

class TestBifurcationAppears:
    def _make_pair(self):
        """G1: straight A--B. G2: A--BP--B with extra branch BP--C."""
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 20, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "BP", 10, 0, 0)
        _make_node(g2, "B2", 20, 0, 0)
        _make_node(g2, "C", 10, 10, 0)
        _make_edge(g2, "A2", "BP")
        _make_edge(g2, "BP", "B2")
        _make_edge(g2, "BP", "C")
        return g1, g2

    def test_split_detected(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        # A--B should match as split to A2--BP--B2
        assert _count_splits(r) == 1

    def test_new_branch_unmatched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        # BP--C should be surplus
        assert len(r.unmatched_edges_g2) >= 1


# ---------------------------------------------------------------------------
# 17. One endpoint unmatched  (edge dangling into void)
# ---------------------------------------------------------------------------

class TestOneEndpointUnmatched:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 10, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "B2", 100, 100, 100)  # far away
        _make_edge(g2, "A2", "B2")
        return g1, g2

    def test_edge_unmatched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        # B and B2 are too far apart -> edge can't match
        assert len(r.edge_pairs) == 0

    def test_partial_node_match(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert "A" in r.node_pairs
        assert "B" in r.unmatched_nodes_g1


# ---------------------------------------------------------------------------
# 18. Both endpoints unmatched
# ---------------------------------------------------------------------------

class TestBothEndpointsUnmatched:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 10, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "C", 200, 200, 200)
        _make_node(g2, "D", 210, 200, 200)
        _make_edge(g2, "C", "D")
        return g1, g2

    def test_nothing_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 0
        assert len(r.edge_pairs) == 0

    def test_both_in_surplus(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert r.surplus_graph.number_of_edges() == 2


# ---------------------------------------------------------------------------
# 19. Rerouted edge  (same endpoints, but spatially different path)
# ---------------------------------------------------------------------------

class TestReroutedEdge:
    def _make_pair(self):
        """
        Both graphs have edges A--B, but g2 stores a curved coords_list
        that deviates from the straight line.
        """
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 20, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "B2", 20, 0, 0)
        # Curved path through (10,10,0)
        coords = [[0, 0, 0], [5, 5, 0], [10, 10, 0], [15, 5, 0], [20, 0, 0]]
        _make_edge(g2, "A2", "B2", coords_list=str(coords))
        return g1, g2

    def test_edges_paired_by_endpoints(self):
        """Endpoint-based matching still pairs these edges."""
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.edge_pairs) == 1


# ---------------------------------------------------------------------------
# 20. Large asymmetric graphs  (g2 much bigger than g1)
# ---------------------------------------------------------------------------

class TestAsymmetricSizes:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_line(g1, "s", [(0, 0, 0), (10, 0, 0)])

        g2 = nx.Graph()
        _make_line(g2, "t", [(0, 0, 0), (10, 0, 0)])
        # Add many extra nodes far away
        for i in range(20):
            _make_node(g2, f"extra_{i}", 50 + i * 5, 50, 50)
        for i in range(19):
            _make_edge(g2, f"extra_{i}", f"extra_{i + 1}")
        return g1, g2

    def test_shared_part_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.node_pairs) == 2
        assert len(r.edge_pairs) == 1

    def test_extra_nodes_all_surplus(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0)
        assert len(r.unmatched_nodes_g2) == 20


# ---------------------------------------------------------------------------
# 21. Hungarian does not steal matches  (augmented matrix correctness)
# ---------------------------------------------------------------------------

class TestHungarianNoStealing:
    """
    Regression test for the original bug: without augmentation, the Hungarian
    algorithm on a rectangular cost matrix would force all min(n1,n2) nodes
    into pairings, stealing good local matches to improve global total cost.
    """

    def _make_pair(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        # Cluster of 3 well-matched pairs
        for i in range(3):
            _make_node(g1, f"A{i}", i * 20, 0, 0)
            _make_node(g2, f"B{i}", i * 20 + 0.5, 0, 0)  # offset 0.5
        # Extra nodes in g2 that are far away
        for i in range(5):
            _make_node(g2, f"far_{i}", 200 + i * 10, 200, 200)
        return g1, g2

    def test_close_pairs_all_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 3
        # Verify the correct pairings (closest nodes)
        for i in range(3):
            assert f"A{i}" in r.node_pairs

    def test_far_nodes_not_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        for i in range(5):
            assert f"far_{i}" in r.unmatched_nodes_g2


# ---------------------------------------------------------------------------
# 22. Surplus graph structure  (correct prefixing and source tags)
# ---------------------------------------------------------------------------

class TestSurplusGraphStructure:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "X", 100, 100, 100)  # unmatched
        _make_edge(g1, "A", "X")

        g2 = nx.Graph()
        _make_node(g2, "B", 0, 0, 0)
        _make_node(g2, "Y", 200, 200, 200)  # unmatched
        _make_edge(g2, "B", "Y")
        return g1, g2

    def test_nodes_prefixed(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        surplus_nodes = set(r.surplus_graph.nodes())
        # Unmatched nodes should be prefixed
        assert "g1_X" in surplus_nodes
        assert "g2_Y" in surplus_nodes

    def test_source_tags(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        for n, attr in r.surplus_graph.nodes(data=True):
            assert "source" in attr
            assert attr["source"] in ("g1", "g2")

    def test_edge_source_tags(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        for u, v, attr in r.surplus_graph.edges(data=True):
            assert "source" in attr


# ---------------------------------------------------------------------------
# 23. Difference graph structure  (correct attributes)
# ---------------------------------------------------------------------------

class TestDifferenceGraphStructure:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 10, 0, 0)
        _make_edge(g1, "A", "B", radius_avg=0.05, length=10.0)

        g2 = nx.Graph()
        _make_node(g2, "C", 1, 0, 0)
        _make_node(g2, "D", 11, 0, 0)
        _make_edge(g2, "C", "D", radius_avg=0.06, length=10.0)
        return g1, g2

    def test_paired_node_attr(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        for n, attr in r.difference_graph.nodes(data=True):
            assert "paired_node_g2" in attr
            assert "X_diff" in attr

    def test_coordinate_diff(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        # Node A at (0,0,0) paired with C at (1,0,0)
        a_attr = r.difference_graph.nodes["A"]
        assert a_attr["X_diff"] == pytest.approx(1.0)

    def test_paired_edge_attr(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        for u, v, attr in r.difference_graph.edges(data=True):
            assert "paired_edge_g2" in attr


# ---------------------------------------------------------------------------
# 24. detect_splits=False  (skip path-based matching)
# ---------------------------------------------------------------------------

class TestSplitDetectionDisabled:
    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 20, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "M", 10, 0, 0)
        _make_node(g2, "B2", 20, 0, 0)
        _make_edge(g2, "A2", "M")
        _make_edge(g2, "M", "B2")
        return g1, g2

    def test_no_splits_when_disabled(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0, detect_splits=False)
        assert len(r.edge_splits) == 0

    def test_edges_remain_unmatched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0, detect_splits=False)
        assert len(r.unmatched_edges_g1) == 1
        assert len(r.unmatched_edges_g2) == 2


# ---------------------------------------------------------------------------
# 25. overlap_threshold  (high threshold rejects weak overlaps)
# ---------------------------------------------------------------------------

class TestOverlapThreshold:
    def _make_pair(self):
        """Split where the multi-edge path deviates from the single edge."""
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 20, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "A2", 0, 0, 0)
        _make_node(g2, "M", 10, 15, 0)   # deviated midpoint
        _make_node(g2, "B2", 20, 0, 0)
        _make_edge(g2, "A2", "M")
        _make_edge(g2, "M", "B2")
        return g1, g2

    def test_low_threshold_accepts(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0,
                           overlap_threshold=0.1, path_distance_threshold=20.0)
        assert _count_splits(r) == 1

    def test_high_threshold_rejects(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=2.0,
                           overlap_threshold=0.99, path_distance_threshold=1.0)
        assert _count_splits(r) == 0


# ---------------------------------------------------------------------------
# 26. MultiGraph input  (defensive conversion)
# ---------------------------------------------------------------------------

class TestMultiGraphInput:
    def test_multigraph_converted(self):
        mg1 = nx.MultiGraph()
        _make_node(mg1, "A", 0, 0, 0)
        _make_node(mg1, "B", 10, 0, 0)
        mg1.add_edge("A", "B", length=10.0, radius_avg=0.05,
                     radius_max=0.07, radius_min=0.03, tortuosity=1.0,
                     volume=0.001, surface_area=0.01)

        mg2 = nx.MultiGraph()
        _make_node(mg2, "C", 0, 0, 0)
        _make_node(mg2, "D", 10, 0, 0)
        mg2.add_edge("C", "D", length=10.0, radius_avg=0.05,
                     radius_max=0.07, radius_min=0.03, tortuosity=1.0,
                     volume=0.001, surface_area=0.01)

        r = compare_graphs(mg1, mg2, max_node_distance=2.0)
        assert len(r.node_pairs) == 2
        assert len(r.edge_pairs) == 1


# ---------------------------------------------------------------------------
# 27. Complex realistic topology  (tree with multiple bifurcations)
# ---------------------------------------------------------------------------

class TestComplexTree:
    """
    G1: Root -> BP1 -> {Leaf1, BP2 -> {Leaf2, Leaf3}}
    G2: Same tree, but BP2 shifted and one extra spur off Leaf3.
    """

    def _make_pair(self):
        g1 = nx.Graph()
        _make_node(g1, "root", 0, 0, 0)
        _make_node(g1, "bp1", 10, 0, 0)
        _make_node(g1, "leaf1", 10, 10, 0)
        _make_node(g1, "bp2", 20, 0, 0)
        _make_node(g1, "leaf2", 20, 10, 0)
        _make_node(g1, "leaf3", 30, 0, 0)
        _make_edge(g1, "root", "bp1")
        _make_edge(g1, "bp1", "leaf1")
        _make_edge(g1, "bp1", "bp2")
        _make_edge(g1, "bp2", "leaf2")
        _make_edge(g1, "bp2", "leaf3")

        g2 = nx.Graph()
        _make_node(g2, "root2", 0, 0, 0)
        _make_node(g2, "bp1_2", 10, 0, 0)
        _make_node(g2, "leaf1_2", 10, 10, 0)
        _make_node(g2, "bp2_2", 22, 0, 0)  # shifted by 2
        _make_node(g2, "leaf2_2", 20, 10, 0)
        _make_node(g2, "leaf3_2", 30, 0, 0)
        _make_node(g2, "spur", 35, 0, 0)   # extra
        _make_edge(g2, "root2", "bp1_2")
        _make_edge(g2, "bp1_2", "leaf1_2")
        _make_edge(g2, "bp1_2", "bp2_2")
        _make_edge(g2, "bp2_2", "leaf2_2")
        _make_edge(g2, "bp2_2", "leaf3_2")
        _make_edge(g2, "leaf3_2", "spur")
        return g1, g2

    def test_all_original_nodes_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 6

    def test_spur_is_surplus(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert "spur" in r.unmatched_nodes_g2

    def test_most_edges_matched(self):
        g1, g2 = self._make_pair()
        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.edge_pairs) == 5  # all 5 original edges


# ---------------------------------------------------------------------------
# 28. edge_in_set / sample_edge_path unit tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_edge_in_set_forward(self):
        assert edge_in_set(("A", "B"), {("A", "B")})

    def test_edge_in_set_reverse(self):
        assert edge_in_set(("B", "A"), {("A", "B")})

    def test_edge_in_set_missing(self):
        assert not edge_in_set(("A", "C"), {("A", "B")})

    def test_sample_edge_path_shape(self):
        g = nx.Graph()
        _make_node(g, "A", 0, 0, 0)
        _make_node(g, "B", 10, 0, 0)
        _make_edge(g, "A", "B")
        path = sample_edge_path(g, ("A", "B"), n_samples=10)
        assert path.shape == (10, 3)
        np.testing.assert_array_almost_equal(path[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(path[-1], [10, 0, 0])

    def test_compute_edge_overlap_identical(self):
        pts = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        assert compute_edge_overlap(pts, pts) == pytest.approx(1.0)

    def test_compute_edge_overlap_distant(self):
        p1 = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        p2 = np.array([[0, 100, 0], [10, 100, 0]], dtype=float)
        assert compute_edge_overlap(p1, p2, distance_threshold=3.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 29. Parallel vessels  (nearby vessels representing the same structure)
# ---------------------------------------------------------------------------

class TestParallelVessels:
    """
    Parallel vessels occur when two segmentations trace the same physical
    vessel at slightly different lateral positions.  Endpoints should be
    close enough to pair, so the edge also pairs.
    """

    def test_simple_parallel_matched(self):
        """Single vessel offset laterally by 2 units — well within threshold."""
        g1 = nx.Graph()
        _make_line(g1, "a", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])

        g2 = nx.Graph()
        _make_line(g2, "b", [(0, 2, 0), (10, 2, 0), (20, 2, 0)])

        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 3
        assert len(r.edge_pairs) == 2
        assert len(r.unmatched_edges_g1) == 0
        assert len(r.unmatched_edges_g2) == 0

    def test_parallel_too_far_unmatched(self):
        """Same vessel offset by 10 units — beyond threshold of 5."""
        g1 = nx.Graph()
        _make_line(g1, "a", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])

        g2 = nx.Graph()
        _make_line(g2, "b", [(0, 10, 0), (10, 10, 0), (20, 10, 0)])

        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 0
        assert len(r.edge_pairs) == 0

    def test_two_parallel_pairs_no_cross_match(self):
        """
        Both graphs have two parallel vessels separated by 15 units.
        Each vessel is offset 2 units between g1/g2.
        The Hungarian algorithm must pair vessel-1 with vessel-1 and
        vessel-2 with vessel-2, not cross-assign them.

        G1:  vessel_a  y=0    o----o----o
             vessel_b  y=15   o----o----o

        G2:  vessel_a  y=2    o----o----o    (offset +2 from g1 vessel_a)
             vessel_b  y=17   o----o----o    (offset +2 from g1 vessel_b)
        """
        g1 = nx.Graph()
        _make_line(g1, "a", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])
        _make_line(g1, "b", [(0, 15, 0), (10, 15, 0), (20, 15, 0)])

        g2 = nx.Graph()
        _make_line(g2, "c", [(0, 2, 0), (10, 2, 0), (20, 2, 0)])
        _make_line(g2, "d", [(0, 17, 0), (10, 17, 0), (20, 17, 0)])

        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 6
        assert len(r.edge_pairs) == 4

        # Verify correct pairing (no cross-assignment):
        # a_0 (y=0) must pair with c_0 (y=2), not d_0 (y=17)
        assert r.node_pairs["a_0"] == "c_0"
        assert r.node_pairs["b_0"] == "d_0"

    def test_close_parallel_pairs_no_cross_match(self):
        """
        Two parallel vessels only 6 units apart.  Each offset by 2 between
        graphs.  Vessels are close enough that a naive algorithm might
        cross-match, but correct 1:1 assignment should not.

        G1:  vessel_a  y=0   o----o----o
             vessel_b  y=6   o----o----o

        G2:  vessel_a  y=2   o----o----o
             vessel_b  y=8   o----o----o
        """
        g1 = nx.Graph()
        _make_line(g1, "a", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])
        _make_line(g1, "b", [(0, 6, 0), (10, 6, 0), (20, 6, 0)])

        g2 = nx.Graph()
        _make_line(g2, "c", [(0, 2, 0), (10, 2, 0), (20, 2, 0)])
        _make_line(g2, "d", [(0, 8, 0), (10, 8, 0), (20, 8, 0)])

        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 6
        assert len(r.edge_pairs) == 4
        # a_i (y=0) must pair with c_i (y=2), not d_i (y=8)
        for i in range(3):
            assert r.node_pairs[f"a_{i}"] == f"c_{i}"
            assert r.node_pairs[f"b_{i}"] == f"d_{i}"

    def test_parallel_3d_offset(self):
        """Parallel vessels offset in Z (depth), typical in volumetric data."""
        g1 = nx.Graph()
        _make_line(g1, "a", [(0, 0, 0), (10, 0, 0), (20, 0, 0)])

        g2 = nx.Graph()
        _make_line(g2, "b", [(0, 0, 3), (10, 0, 3), (20, 0, 3)])

        r = compare_graphs(g1, g2, max_node_distance=5.0)
        assert len(r.node_pairs) == 3
        assert len(r.edge_pairs) == 2

    def test_parallel_with_jitter(self):
        """
        Parallel vessel with per-node random jitter — simulates noisy
        segmentation of the same vessel.
        """
        rng = np.random.default_rng(123)
        jitter = 1.5
        base_coords = [(i * 10, 0, 0) for i in range(5)]

        g1 = nx.Graph()
        g2 = nx.Graph()
        for i, (x, y, z) in enumerate(base_coords):
            _make_node(g1, f"a_{i}", x, y, z)
            dx, dy, dz = rng.uniform(-jitter, jitter, 3)
            _make_node(g2, f"b_{i}", x + dx, y + 2 + dy, z + dz)
        for i in range(4):
            _make_edge(g1, f"a_{i}", f"a_{i+1}")
            _make_edge(g2, f"b_{i}", f"b_{i+1}")

        r = compare_graphs(g1, g2, max_node_distance=7.0)
        assert len(r.node_pairs) == 5
        assert len(r.edge_pairs) == 4

    def test_parallel_different_sampling(self):
        """
        Same vessel, but g2 has more intermediate nodes (finer sampling).
        G1: A------B  (one long edge)
        G2: C--M--D   (two shorter edges, same spatial path)
        The edge should be detected as a split.
        """
        g1 = nx.Graph()
        _make_node(g1, "A", 0, 0, 0)
        _make_node(g1, "B", 20, 0, 0)
        _make_edge(g1, "A", "B")

        g2 = nx.Graph()
        _make_node(g2, "C", 0, 2, 0)   # offset 2 in Y
        _make_node(g2, "M", 10, 2, 0)
        _make_node(g2, "D", 20, 2, 0)
        _make_edge(g2, "C", "M")
        _make_edge(g2, "M", "D")

        r = compare_graphs(g1, g2, max_node_distance=5.0,
                           path_distance_threshold=5.0)
        # Endpoints A<->C and B<->D are paired; edge should be a split
        assert "A" in r.node_pairs
        assert "B" in r.node_pairs
        assert _count_splits(r) == 1
