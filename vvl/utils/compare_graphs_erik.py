import ast
import pickle

import networkx as nx
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------


def load_graph(input_path):
    """Load a pickled igraph, convert to undirected NetworkX graph."""
    with open(input_path, "rb") as f:
        g = pickle.load(f)
        g = nx.Graph(g.to_networkx())
    return g


def save_graph(g, output_path):
    """Write graph to GraphML, stripping attributes that don't serialise well."""
    attrs_to_remove_nodes = ["v_coords"]
    attrs_to_remove_edges = [
        "radii_list",
        "coords_list",
        "original_edge_positions",
        "original_edge_paths",
    ]

    for node in g.nodes():
        for attr in attrs_to_remove_nodes:
            if attr in g.nodes[node]:
                del g.nodes[node][attr]

    for u, v in g.edges():
        for attr in attrs_to_remove_edges:
            if attr in g.edges[u, v]:
                del g.edges[u, v][attr]

    nx.write_graphml(g, output_path)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def get_node_coords(graph: nx.Graph, node_id) -> np.ndarray:
    """Extract XYZ coordinates from a node."""
    d = graph.nodes[node_id]
    return np.array([float(d["X"]), float(d["Y"]), float(d["Z"])])


def get_all_node_coords(graph: nx.Graph) -> tuple[list, np.ndarray]:
    """Get all node IDs and their coordinates as arrays."""
    node_ids = list(graph.nodes())
    coords = np.array([get_node_coords(graph, n) for n in node_ids])
    return node_ids, coords


def _get_pos(graph: nx.Graph) -> dict[int, np.ndarray]:
    """Return a dict mapping each node to its 3D position."""
    return {n: get_node_coords(graph, n) for n in graph.nodes()}


# ---------------------------------------------------------------------------
# Edge geometry helpers
# ---------------------------------------------------------------------------


def get_edge_direction(G, edge) -> np.ndarray:
    """Get the unit 3D direction vector of an edge (zero-length edges return zero vector)."""
    u, v = edge
    direction = get_node_coords(G, v) - get_node_coords(G, u)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


def angular_similarity(G1, edge1, G2, edge2) -> float:
    """
    Compute angular similarity between two edges.
    Returns angle in degrees (0 = parallel, 90 = perpendicular).
    Ignores direction (treats edges as undirected).
    """
    dir1 = get_edge_direction(G1, edge1)
    dir2 = get_edge_direction(G2, edge2)
    cos_angle = np.abs(np.clip(np.dot(dir1, dir2), -1, 1))
    return float(np.degrees(np.arccos(cos_angle)))


def get_edge_points(G, edge) -> np.ndarray:
    """Get the coordinate list stored on an edge, parsing from string if needed."""
    coords = G.edges[edge]["coords_list"]
    if isinstance(coords, str):
        coords = ast.literal_eval(coords)
    return np.array(coords)


def get_edge_length(g, u, v) -> float:
    """Get physical length of an edge (sum of segment lengths, or endpoint distance as fallback)."""
    edge_data = g.edges[u, v]
    if "coords_list" in edge_data:
        coords = edge_data["coords_list"]
        if isinstance(coords, str):
            coords = ast.literal_eval(coords)
        coords = np.array(coords)
        if len(coords) >= 2:
            return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))

    p1 = get_node_coords(g, u)
    p2 = get_node_coords(g, v)
    return float(np.linalg.norm(p2 - p1))


def _get_edge_attr_float(graph: nx.Graph, u, v, attr: str, default: float = 0.0) -> float:
    """Get a float edge attribute, flipping edge direction if necessary."""
    if not graph.has_edge(u, v):
        u, v = v, u
    return float(graph.edges[u, v].get(attr, default))


# ---------------------------------------------------------------------------
# Shared match-analysis helpers (used by both print_match_summary and get_match_stats)
# ---------------------------------------------------------------------------


def _is_matched(edge: tuple, matched_set: set) -> bool:
    """Check whether an edge (or its reverse) is in matched_set."""
    return edge in matched_set or (edge[1], edge[0]) in matched_set


def _aggregate_edge_features(graph: nx.Graph, edges) -> tuple[float, float, float]:
    """
    Aggregate length, volume, and length-weighted radius over a set of edges.

    Returns (total_length, total_volume, weighted_radius).
    """
    lengths, volumes, radii = [], [], []
    for e in edges:
        u, v = e
        if not graph.has_edge(u, v):
            u, v = v, u
        d = graph.edges[u, v]
        lengths.append(float(d.get("length", 0.0)))
        volumes.append(float(d.get("volume", 0.0)))
        radii.append(float(d.get("radius_avg", 0.0)))

    total_len = sum(lengths)
    total_vol = sum(volumes)
    w_radius = (
        sum(r * l for r, l in zip(radii, lengths)) / total_len
        if total_len > 0
        else 0.0
    )
    return total_len, total_vol, w_radius


def _collect_edge_geometry(graph, edges, pos):
    """Collect line-segment points and VTK-style line indices for a set of edges."""
    points, lines = [], []
    for edge in edges:
        u, v = edge[0], edge[1]
        if u not in pos or v not in pos:
            continue
        p1, p2 = pos[u], pos[v]
        start_idx = len(points)
        points.extend([p1, p2])
        lines.extend([2, start_idx, start_idx + 1])
    return points, lines


# ---------------------------------------------------------------------------
# Spatial matching building blocks
# ---------------------------------------------------------------------------


def find_nearby_nodes(edge_coords: np.ndarray, graph: nx.Graph, max_distance: float) -> set:
    """
    Find all nodes in *graph* that are within *max_distance* of any point
    in *edge_coords* (shape (N, 3)).
    """
    if len(edge_coords) == 0:
        return set()

    edge_tree = KDTree(edge_coords)

    nearby_nodes = set()
    for node in graph.nodes():
        node_coord = get_node_coords(graph, node)
        dist, _ = edge_tree.query(node_coord)
        if dist <= max_distance:
            nearby_nodes.add(node)

    return nearby_nodes


def build_candidate_subgraph(graph: nx.Graph, candidate_nodes: set) -> nx.Graph:
    """Build a subgraph containing only edges where BOTH endpoints are in *candidate_nodes*."""
    subgraph = nx.Graph()

    for node in candidate_nodes:
        if node in graph.nodes:
            subgraph.add_node(node, **graph.nodes[node])

    for u, v in graph.edges():
        if u in candidate_nodes and v in candidate_nodes:
            subgraph.add_edge(u, v, **graph.edges[u, v])

    return subgraph


# ---------------------------------------------------------------------------
# Path-finding in subgraphs
# ---------------------------------------------------------------------------


def path_length(path, subgraph) -> float:
    """Total physical length of a path through *subgraph*."""
    return sum(
        get_edge_length(subgraph, path[i], path[i + 1])
        for i in range(len(path) - 1)
    )


def find_longest_path_in_subgraph(subgraph: nx.Graph) -> list:
    """Find the longest (by physical length) path in *subgraph* via DFS from endpoints."""
    if len(subgraph.nodes) == 0:
        return []
    if len(subgraph.nodes) == 1:
        return list(subgraph.nodes)

    endpoints = [n for n in subgraph.nodes if subgraph.degree(n) == 1]
    if not endpoints:
        endpoints = [list(subgraph.nodes)[0]]

    longest = []
    longest_len = 0

    for start in endpoints:
        stack = [(start, [start], {start})]
        while stack:
            node, path, visited = stack.pop()
            neighbors = [n for n in subgraph.neighbors(node) if n not in visited]

            if not neighbors:
                plen = path_length(path, subgraph)
                if plen > longest_len:
                    longest = path
                    longest_len = plen
            else:
                for neighbor in neighbors:
                    stack.append((neighbor, path + [neighbor], visited | {neighbor}))

    return longest


def find_length_matched_path_in_subgraph(
    subgraph: nx.Graph,
    start: int,
    end: tuple,
    source_len: float,
) -> list:
    """
    Find the path whose length best matches *source_len*, starting from *start*.

    If the *end* node (given as ``(node_id, distance)``) is close enough,
    the shortest path to it is tried first.
    """
    if len(subgraph.nodes) == 0:
        return []
    if len(subgraph.nodes) == 1:
        return list(subgraph.nodes)

    best_path = []
    best_len_diff = np.inf

    end_node, end_dist = end
    if end_dist < 5:
        try:
            best_path = nx.shortest_path(subgraph, source=start, target=end_node, weight="length")
        except nx.exception.NetworkXNoPath:
            pass

    if len(best_path) < 2:
        stack = [(start, [start], {start})]
        while stack:
            node, path, visited = stack.pop()
            neighbors = [n for n in subgraph.neighbors(node) if n not in visited]

            plen = path_length(path, subgraph)
            if abs(plen - source_len) < best_len_diff:
                best_path = path
                best_len_diff = abs(plen - source_len)

            for neighbor in neighbors:
                stack.append((neighbor, path + [neighbor], visited | {neighbor}))

    return best_path


def crop_path_to_endpoints(
    path: list,
    graph: nx.Graph,
    start_coord: np.ndarray,
    end_coord: np.ndarray,
) -> list:
    """Crop *path* so its start/end nodes are closest to the given coordinates."""
    if len(path) <= 1:
        return path

    coords = [get_node_coords(graph, node) for node in path]

    start_idx = int(np.argmin([np.linalg.norm(c - start_coord) for c in coords]))
    end_idx = int(np.argmin([np.linalg.norm(c - end_coord) for c in coords]))

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return path[start_idx : end_idx + 1]


# ---------------------------------------------------------------------------
# Core edge matching
# ---------------------------------------------------------------------------


def get_matched_edges_for_edge(
    edge: tuple,
    g1: nx.Graph,
    g2: nx.Graph,
    max_distance: float = 5.0,
) -> set:
    """
    Find matching edges in *g2* for a given edge in *g1* using node-based matching.

    Algorithm:
    1. Get coords_list of the edge in g1
    2. Find all nodes in g2 within max_distance of any point in coords_list
    3. Build a subgraph of g2 with only those candidate nodes
    4. Find the best length-matched path in that subgraph
    5. Crop the path so start/end are closest to original edge endpoints
    6. Reject if angular similarity exceeds 30 degrees
    """
    u, v = edge
    source_len = get_edge_length(g1, u, v)
    start_coord = get_node_coords(g1, u)
    end_coord = get_node_coords(g1, v)
    edge_coords = get_edge_points(g1, edge)

    candidate_nodes = find_nearby_nodes(edge_coords, g2, max_distance)
    if len(candidate_nodes) < 2:
        return set()

    subgraph = build_candidate_subgraph(g2, candidate_nodes)
    if len(subgraph.edges) == 0:
        return set()

    # Find closest candidate nodes to the source edge endpoints
    def _closest_candidate(target_coord):
        return min(
            ((node, np.linalg.norm(get_node_coords(g2, node) - target_coord))
             for node in candidate_nodes),
            key=lambda x: x[1],
        )

    start_match = _closest_candidate(start_coord)
    end_match = _closest_candidate(end_coord)

    # Start DFS from whichever endpoint had a closer match
    if start_match[1] > end_match[1]:
        best_path = find_length_matched_path_in_subgraph(subgraph, end_match[0], start_match, source_len)
    else:
        best_path = find_length_matched_path_in_subgraph(subgraph, start_match[0], end_match, source_len)

    if len(best_path) < 2:
        return set()

    cropped_path = crop_path_to_endpoints(best_path, g2, start_coord, end_coord)
    if len(cropped_path) < 2:
        return set()

    # TODO: Frechet distance
    if angular_similarity(g1, (u, v), g2, (cropped_path[0], cropped_path[-1])) > 30:
        return set()

    return {
        (cropped_path[i], cropped_path[i + 1])
        for i in range(len(cropped_path) - 1)
    }


def get_node_dists_from_edge(e1, g1, g2, edge_match) -> float:
    """Sum of minimum distances from source-edge endpoints to all matched-edge nodes."""
    start_coord = get_node_coords(g1, e1[0])
    end_coord = get_node_coords(g1, e1[1])

    match_nodes = {node for sublist in edge_match for node in sublist}
    match_coords = [get_node_coords(g2, node) for node in match_nodes]

    start_dists = [np.linalg.norm(c - start_coord) for c in match_coords]
    end_dists = [np.linalg.norm(c - end_coord) for c in match_coords]
    return min(start_dists) + min(end_dists)


# ---------------------------------------------------------------------------
# Full matching pipeline
# ---------------------------------------------------------------------------


def run_matching(g1: nx.Graph, g2: nx.Graph, distance: float = 13.0) -> list:
    """
    Run the full bidirectional matching pipeline between g1 and g2.

    Returns
    -------
    list of (g1_edges_set, g2_edges_set) tuples
    """
    matched_es1_dict = {}
    matched_es2_dict = {}

    for e1 in tqdm(g1.edges, desc="Matching g1->g2"):
        matched = get_matched_edges_for_edge(e1, g1, g2, max_distance=distance)
        if matched:
            matched_es1_dict[e1] = matched

    for e2 in tqdm(g2.edges, desc="Matching g2->g1"):
        matched = get_matched_edges_for_edge(e2, g2, g1, max_distance=distance)
        if matched:
            matched_es2_dict[e2] = matched

    # Combine: all g1->g2 matches, plus multi-edge g2->g1 matches
    final_matches = []
    for e1_key, e1_match in matched_es1_dict.items():
        final_matches.append(({e1_key}, e1_match))
    for e2_key, e2_match in matched_es2_dict.items():
        if len(e2_match) >= 2:
            final_matches.append((e2_match, {e2_key}))

    return final_matches


# ---------------------------------------------------------------------------
# Statistics and reporting
# ---------------------------------------------------------------------------


def print_match_summary(g1: nx.Graph, g2: nx.Graph, final_matches: list, filter_length: float = 0.0):
    """
    Print per-match and aggregate errors for length, volume, and radius_avg.

    Errors are g2 - g1 (absolute) and (g2 - g1) / g1 * 100 (relative %).
    """
    rows = []
    for i, (g1_edges, g2_edges) in enumerate(final_matches):
        l1, v1, r1 = _aggregate_edge_features(g1, g1_edges)
        if l1 < filter_length:
            continue
        l2, v2, r2 = _aggregate_edge_features(g2, g2_edges)
        rows.append(
            {
                "i": i,
                "n_g1": len(g1_edges),
                "n_g2": len(g2_edges),
                "len_g1": l1,  "len_g2": l2,
                "len_abs": l2 - l1,
                "len_rel": (l2 - l1) / l1 * 100 if l1 > 0 else float("nan"),
                "vol_g1": v1,  "vol_g2": v2,
                "vol_abs": v2 - v1,
                "vol_rel": (v2 - v1) / v1 * 100 if v1 > 0 else float("nan"),
                "rad_g1": r1,  "rad_g2": r2,
                "rad_abs": r2 - r1,
                "rad_rel": (r2 - r1) / r1 * 100 if r1 > 0 else float("nan"),
            }
        )

    if not rows:
        print("No matches to summarize.")
        return

    # Per-match table
    rows.sort(key=lambda r: r["len_g1"])
    header = (
        f"{'#':>5}  {'g1e':>4}{'g2e':>4}"
        f"  {'len_g1':>8}{'len_g2':>8}{'len_err':>9}{'len_%':>7}"
        f"  {'vol_g1':>10}{'vol_g2':>10}{'vol_err':>11}{'vol_%':>7}"
        f"  {'rad_g1':>7}{'rad_g2':>7}{'rad_err':>8}{'rad_%':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['i']:>5}  {r['n_g1']:>4}{r['n_g2']:>4}"
            f"  {r['len_g1']:>8.1f}{r['len_g2']:>8.1f}{r['len_abs']:>+9.1f}{r['len_rel']:>+7.1f}"
            f"  {r['vol_g1']:>10.1f}{r['vol_g2']:>10.1f}{r['vol_abs']:>+11.1f}{r['vol_rel']:>+7.1f}"
            f"  {r['rad_g1']:>7.2f}{r['rad_g2']:>7.2f}{r['rad_abs']:>+8.3f}{r['rad_rel']:>+7.1f}"
        )

    # Aggregate stats
    def _stats(key):
        vals = np.array([r[key] for r in rows])
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return dict(mean=float("nan"), median=float("nan"), std=float("nan"))
        return dict(mean=np.mean(finite).item(), median=np.median(finite).item(), std=np.std(finite).item())

    print()
    print(f"{'':20} {'mean':>10} {'median':>10} {'std':>10}")
    print("-" * 52)
    for label, key in [
        ("length abs err",   "len_abs"),
        ("length rel err %", "len_rel"),
        ("volume abs err",   "vol_abs"),
        ("volume rel err %", "vol_rel"),
        ("radius abs err",   "rad_abs"),
        ("radius rel err %", "rad_rel"),
    ]:
        s = _stats(key)
        print(f"{label:20} {s['mean']:>+10.3f} {s['median']:>+10.3f} {s['std']:>10.3f}")
    print(f"\nTotal matches: {len(rows)}")

    # Unmatched edge stats
    matched_edges_g1 = set()
    matched_edges_g2 = set()
    for g1_edges, g2_edges in final_matches:
        matched_edges_g1.update(g1_edges)
        matched_edges_g2.update(g2_edges)

    def _unmatched_edges(graph, matched_set):
        return [
            e for e in graph.edges()
            if not _is_matched(e, matched_set)
            and _get_edge_attr_float(graph, e[0], e[1], "length") >= filter_length
        ]

    def _edge_arrays(graph, edges):
        lengths, volumes, radii = [], [], []
        for u, v in edges:
            if not graph.has_edge(u, v):
                u, v = v, u
            d = graph.edges[u, v]
            lengths.append(float(d.get("length", 0.0)))
            volumes.append(float(d.get("volume", 0.0)))
            radii.append(float(d.get("radius_avg", 0.0)))
        return np.array(lengths), np.array(volumes), np.array(radii)

    def _print_unmatched_stats(label, graph, edges):
        if not edges:
            print(f"\n{label}: 0 unmatched edges")
            return
        lens, vols, rads = _edge_arrays(graph, edges)
        print(f"\n{label}: {len(edges)} unmatched edges")
        print(f"  {'':16} {'mean':>10} {'median':>10} {'std':>10} {'total':>12}")
        print(f"  {'-' * 60}")
        for name, arr in [("length", lens), ("volume", vols), ("radius_avg", rads)]:
            total = f"{arr.sum():>12.1f}" if name != "radius_avg" else f"{'--':>12}"
            print(f"  {name:16} {arr.mean():>10.3f} {np.median(arr):>10.3f} {arr.std():>10.3f} {total}")

    _print_unmatched_stats("G1 unmatched", g1, _unmatched_edges(g1, matched_edges_g1))
    _print_unmatched_stats("G2 unmatched", g2, _unmatched_edges(g2, matched_edges_g2))


def get_match_stats(g1: nx.Graph, g2: nx.Graph, final_matches: list, filter_length: float = 0.0) -> dict:
    """
    Compute summary statistics for a set of final_matches.

    Returns a flat dict with keys covering:
      - matching overview (counts, unmatched totals)
      - per-feature error stats (mean/median abs and rel for length, volume, radius_avg)
    """
    len_abs, len_rel = [], []
    vol_abs, vol_rel = [], []
    rad_abs, rad_rel = [], []
    matched_g1, matched_g2 = set(), set()

    for g1_edges, g2_edges in final_matches:
        matched_g1.update(g1_edges)
        matched_g2.update(g2_edges)
        l1, v1, r1 = _aggregate_edge_features(g1, g1_edges)
        if l1 < filter_length:
            continue
        l2, v2, r2 = _aggregate_edge_features(g2, g2_edges)
        len_abs.append(l2 - l1)
        len_rel.append((l2 - l1) / l1 * 100 if l1 > 0 else float("nan"))
        vol_abs.append(v2 - v1)
        vol_rel.append((v2 - v1) / v1 * 100 if v1 > 0 else float("nan"))
        rad_abs.append(r2 - r1)
        rad_rel.append((r2 - r1) / r1 * 100 if r1 > 0 else float("nan"))

    def _finite_stat(arr, fn):
        a = np.array(arr)
        a = a[np.isfinite(a)]
        return fn(a).item() if len(a) > 0 else float("nan")

    def _unmatched_edges(graph, matched_set):
        return [
            e for e in graph.edges()
            if not _is_matched(e, matched_set)
            and _get_edge_attr_float(graph, e[0], e[1], "length") >= filter_length
        ]

    unmatched_g1 = _unmatched_edges(g1, matched_g1)
    unmatched_g2 = _unmatched_edges(g2, matched_g2)

    def _unmatched_totals(graph, edges):
        """Compute total/mean/std for length and volume of unmatched edges."""
        lengths = [_get_edge_attr_float(graph, u, v, "length") for u, v in edges]
        volumes = [_get_edge_attr_float(graph, u, v, "volume") for u, v in edges]
        arr_len = np.array(lengths) if lengths else np.array([0.0])
        arr_vol = np.array(volumes) if volumes else np.array([0.0])
        return (
            arr_len.sum(), np.mean(arr_len).item(), np.std(arr_len).item(),
            arr_vol.sum(), np.mean(arr_vol).item(), np.std(arr_vol).item(),
        )

    unm_len_total_g1, unm_len_mean_g1, unm_len_std_g1, unm_vol_total_g1, unm_vol_mean_g1, unm_vol_std_g1 = _unmatched_totals(g1, unmatched_g1)
    unm_len_total_g2, unm_len_mean_g2, unm_len_std_g2, unm_vol_total_g2, unm_vol_mean_g2, unm_vol_std_g2 = _unmatched_totals(g2, unmatched_g2)

    return {
        "n_matches":          len(final_matches),
        "n_matched_g1":       len(matched_g1),
        "n_matched_g2":       len(matched_g2),
        "n_unmatched_g1":     len(unmatched_g1),
        "n_unmatched_g2":     len(unmatched_g2),
        "unm_len_total_g1":   unm_len_total_g1,
        "unm_len_total_g2":   unm_len_total_g2,
        "unm_len_mean_g1":    unm_len_mean_g1,
        "unm_len_mean_g2":    unm_len_mean_g2,
        "unm_len_std_g1":     unm_len_std_g1,
        "unm_len_std_g2":     unm_len_std_g2,
        "unm_vol_total_g1":   unm_vol_total_g1,
        "unm_vol_total_g2":   unm_vol_total_g2,
        "unm_vol_mean_g1":    unm_vol_mean_g1,
        "unm_vol_mean_g2":    unm_vol_mean_g2,
        "unm_vol_std_g1":     unm_vol_std_g1,
        "unm_vol_std_g2":     unm_vol_std_g2,
        "mean_len_abs":       _finite_stat(len_abs, np.mean),
        "median_len_abs":     _finite_stat(len_abs, np.median),
        "mean_len_rel":       _finite_stat(len_rel, np.mean),
        "median_len_rel":     _finite_stat(len_rel, np.median),
        "mean_vol_abs":       _finite_stat(vol_abs, np.mean),
        "median_vol_abs":     _finite_stat(vol_abs, np.median),
        "mean_vol_rel":       _finite_stat(vol_rel, np.mean),
        "median_vol_rel":     _finite_stat(vol_rel, np.median),
        "mean_rad_abs":       _finite_stat(rad_abs, np.mean),
        "median_rad_abs":     _finite_stat(rad_abs, np.median),
        "mean_rad_rel":       _finite_stat(rad_rel, np.mean),
        "median_rad_rel":     _finite_stat(rad_rel, np.median),
    }


def compare_representations(
    g_gt: nx.Graph,
    comparison_graphs: dict,
    distance: float = 13.0,
    filter_length: float = 0.0,
):
    """
    Compare g_gt against multiple reconstructions and print a side-by-side summary table.

    Parameters
    ----------
    g_gt : nx.Graph
        Ground-truth graph.
    comparison_graphs : dict
        Ordered mapping of label -> nx.Graph, e.g. {"i9": g_i9, "i16": g_i16}.
    distance : float
        Max matching distance (voxels).
    filter_length : float
        Minimum edge length to include in statistics.
    """
    labels = list(comparison_graphs.keys())
    all_stats = {}
    all_matches = {}

    for label, g2 in comparison_graphs.items():
        print(f"\n{'=' * 60}")
        print(f"  Matching GT vs {label}")
        print(f"{'=' * 60}")
        fm = run_matching(g_gt, g2, distance=distance)
        all_matches[label] = (g2, fm)
        all_stats[label] = get_match_stats(g_gt, g2, fm, filter_length=filter_length)

    # Side-by-side table
    col_w = 14
    label_w = 32

    def _header():
        row = f"{'':>{label_w}}"
        for lbl in labels:
            row += f"{'gt vs ' + lbl:>{col_w}}"
        return row

    def _row(name, key, fmt="{:>+.1f}", integer=False):
        row = f"{name:<{label_w}}"
        for lbl in labels:
            val = all_stats[lbl][key]
            if integer:
                row += f"{int(val):>{col_w}}"
            elif np.isfinite(val):
                row += f"{fmt.format(val):>{col_w}}"
            else:
                row += f"{'nan':>{col_w}}"
        return row

    sep = "-" * (label_w + col_w * len(labels))

    print(f"\n{'GT vs reconstruction comparison':^{label_w + col_w * len(labels)}}")
    print("=" * (label_w + col_w * len(labels)))
    print(_header())
    print(sep)

    print("\n--- Matching overview")
    print(_row("Matched pairs",              "n_matches",        integer=True))
    print(_row("Matched G1 edges",           "n_matched_g1",     integer=True))
    print(_row("Matched G2 edges",           "n_matched_g2",     integer=True))
    print(_row("Unmatched G1 edges",         "n_unmatched_g1",   integer=True))
    print(_row("Unmatched G2 edges",         "n_unmatched_g2",   integer=True))
    print(_row("Unmatched G1 total length",  "unm_len_total_g1", fmt="{:.1f}"))
    print(_row("Unmatched G2 total length",  "unm_len_total_g2", fmt="{:.1f}"))
    print(_row("Unmatched G1 mean length",   "unm_len_mean_g1",  fmt="{:.1f}"))
    print(_row("Unmatched G2 mean length",   "unm_len_mean_g2",  fmt="{:.1f}"))
    print(_row("Unmatched G1 std length",    "unm_len_std_g1",   fmt="{:.1f}"))
    print(_row("Unmatched G2 std length",    "unm_len_std_g2",   fmt="{:.1f}"))
    print(_row("Unmatched G1 total volume",  "unm_vol_total_g1", fmt="{:.7f}"))
    print(_row("Unmatched G2 total volume",  "unm_vol_total_g2", fmt="{:.7f}"))
    print(_row("Unmatched G1 mean volume",   "unm_vol_mean_g1",  fmt="{:.7f}"))
    print(_row("Unmatched G2 mean volume",   "unm_vol_mean_g2",  fmt="{:.7f}"))
    print(_row("Unmatched G1 std volume",    "unm_vol_std_g1",   fmt="{:.7f}"))
    print(_row("Unmatched G2 std volume",    "unm_vol_std_g2",   fmt="{:.7f}"))

    print("\n--- Length error (vx)")
    print(_row("  Mean abs",     "mean_len_abs",   fmt="{:>+.2f}"))
    print(_row("  Median abs",   "median_len_abs", fmt="{:>+.2f}"))
    print(_row("  Mean rel %",   "mean_len_rel",   fmt="{:>+.1f}"))
    print(_row("  Median rel %", "median_len_rel", fmt="{:>+.1f}"))

    print("\n--- Volume error")
    print(_row("  Mean abs",     "mean_vol_abs",   fmt="{:>+.2f}"))
    print(_row("  Median abs",   "median_vol_abs", fmt="{:>+.2f}"))
    print(_row("  Mean rel %",   "mean_vol_rel",   fmt="{:>+.1f}"))
    print(_row("  Median rel %", "median_vol_rel", fmt="{:>+.1f}"))

    print("\n--- Radius error (vx)")
    print(_row("  Mean abs",     "mean_rad_abs",   fmt="{:>+.3f}"))
    print(_row("  Median abs",   "median_rad_abs", fmt="{:>+.3f}"))
    print(_row("  Mean rel %",   "mean_rad_rel",   fmt="{:>+.1f}"))
    print(_row("  Median rel %", "median_rad_rel", fmt="{:>+.1f}"))

    print()
    return all_stats, all_matches


# ---------------------------------------------------------------------------
# Graph construction from unmatched edges
# ---------------------------------------------------------------------------


def create_unmatched_edges_graph(
    graph1: nx.Graph,
    graph2: nx.Graph,
    unmatched_edges_g1: set,
    unmatched_edges_g2: set,
) -> nx.Graph:
    """
    Create a graph containing only unmatched edges and their endpoint nodes.

    Nodes and edges are prefixed with their source graph ('g1_' or 'g2_') and
    tagged with a 'source' attribute to indicate origin.
    """
    surplus = nx.Graph()

    sources = [
        ("g1", graph1, unmatched_edges_g1),
        ("g2", graph2, unmatched_edges_g2),
    ]

    for prefix, graph, unmatched_edges in sources:
        for edge in unmatched_edges:
            u, v = edge[0], edge[1]
            for node in (u, v):
                node_id = f"{prefix}_{node}"
                if node_id not in surplus:
                    attr = dict(graph.nodes[node])
                    attr["source"] = prefix
                    attr["original_id"] = node
                    attr["v_coords"] = np.array([float(attr["Z"]), float(attr["Y"]), float(attr["X"])])
                    surplus.add_node(node_id, **attr)

            edge_attr = dict(graph.edges[(u, v)])
            edge_attr["source"] = prefix
            edge_attr["original_edge"] = str((u, v))
            surplus.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **edge_attr)

    return surplus


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _add_edge_tubes(plotter, points, lines, color, label, tube_radius):
    """Add tube-rendered edges to a PyVista plotter if points exist."""
    if points:
        mesh = pv.PolyData(np.array(points))
        mesh.lines = np.array(lines)
        tubes = mesh.tube(radius=tube_radius)
        plotter.add_mesh(tubes, color=color, label=label, pickable=True)


COLORS = {
    "matched_g1":   "#2E8B57",  # Sea green
    "matched_g2":   "#4169E1",  # Royal blue
    "unmatched_g1": "#DC143C",  # Crimson
    "unmatched_g2": "#FF8C00",  # Dark orange
}


def viz_graph(G):
    """Render a graph with edges colored by their 'source' attribute."""
    pos = _get_pos(G)

    g1_points, g1_lines = [], []
    g2_points, g2_lines = [], []

    for u, v in G.edges():
        p1, p2 = pos[u], pos[v]
        source = G.edges[u, v].get("source", "g1")

        if source == "g1":
            start_idx = len(g1_points)
            g1_points.extend([p1, p2])
            g1_lines.extend([2, start_idx, start_idx + 1])
        else:
            start_idx = len(g2_points)
            g2_points.extend([p1, p2])
            g2_lines.extend([2, start_idx, start_idx + 1])

    plotter = pv.Plotter()
    plotter.set_background("white")

    if g1_points:
        g1_mesh = pv.PolyData(np.array(g1_points))
        g1_mesh.lines = np.array(g1_lines)
        plotter.add_mesh(g1_mesh.tube(radius=0.8), color="#E63946", label="Graph 1")

    if g2_points:
        g2_mesh = pv.PolyData(np.array(g2_points))
        g2_mesh.lines = np.array(g2_lines)
        plotter.add_mesh(g2_mesh.tube(radius=0.8), color="#457B9D", label="Graph 2")

    plotter.add_legend()
    plotter.enable_anti_aliasing()
    plotter.show(auto_close=False, interactive_update=True)


def viz_matched_unmatched(
    g1: nx.Graph,
    g2: nx.Graph,
    matched_edges_g1: set,
    matched_edges_g2: set,
    unmatched_edges_g1: set,
    unmatched_edges_g2: set,
    tube_radius: float = 0.8,
):
    """Visualize matched and unmatched edges from two graphs with distinct colors."""
    pos1 = _get_pos(g1)
    pos2 = _get_pos(g2)

    groups = [
        (g1, matched_edges_g1,   pos1, COLORS["matched_g1"],   "G1 Matched"),
        (g2, matched_edges_g2,   pos2, COLORS["matched_g2"],   "G2 Matched"),
        (g1, unmatched_edges_g1, pos1, COLORS["unmatched_g1"], "G1 Unmatched"),
        (g2, unmatched_edges_g2, pos2, COLORS["unmatched_g2"], "G2 Unmatched"),
    ]

    plotter = pv.Plotter()
    plotter.set_background("white")

    for graph, edges, pos, color, label in groups:
        points, lines = _collect_edge_geometry(graph, edges, pos)
        _add_edge_tubes(plotter, points, lines, color, label, tube_radius)

    plotter.add_legend()
    plotter.enable_anti_aliasing()
    plotter.show(auto_close=False, interactive_update=True)


def viz_matched_unmatched_interactive(
    g1: nx.Graph,
    g2: nx.Graph,
    matched_edges_g1: set,
    matched_edges_g2: set,
    unmatched_edges_g1: set,
    unmatched_edges_g2: set,
    matched_es1_dict: dict = None,
    matched_es2_dict: dict = None,
    tube_radius: float = 0.8,
):
    """
    Interactive version of viz_matched_unmatched.
    Click on an edge to print its node IDs, category, and matching info to the console.
    """
    pos1 = _get_pos(g1)
    pos2 = _get_pos(g2)

    # Build lookup: sample start, mid, end of every edge for KDTree nearest-neighbour search
    sample_points = []
    sample_edge_info = []

    edge_groups = [
        (g1, matched_edges_g1,   pos1, "g1", "G1 Matched"),
        (g2, matched_edges_g2,   pos2, "g2", "G2 Matched"),
        (g1, unmatched_edges_g1, pos1, "g1", "G1 Unmatched"),
        (g2, unmatched_edges_g2, pos2, "g2", "G2 Unmatched"),
    ]

    for graph, edges, pos, graph_key, category in edge_groups:
        for edge in edges:
            u, v = edge[0], edge[1]
            if u not in pos or v not in pos:
                continue
            p1, p2 = pos[u], pos[v]
            info = {"category": category, "graph_key": graph_key, "edge": (u, v), "graph": graph}
            for pt in [p1, (p1 + p2) / 2, p2]:
                sample_points.append(pt)
                sample_edge_info.append(info)

    if not sample_points:
        print("No edges to display")
        return

    edge_tree = KDTree(np.array(sample_points))

    # Build matching lookup from both dicts
    match_info = {}
    if matched_es1_dict:
        for e1, e2_set in matched_es1_dict.items():
            match_info[("g1", e1)] = ("matches in g2", e2_set)
    if matched_es2_dict:
        for e2, e1_set in matched_es2_dict.items():
            match_info[("g2", e2)] = ("matches in g1", e1_set)

    def _show_match_detail(source_graph, source_edge, source_key, target_graph, target_edges, target_key, pos_src, pos_tgt):
        """Open a new plotter showing the source edge and its matched target edges."""
        detail = pv.Plotter(title=f"Match detail: {source_key} edge {source_edge}")
        detail.set_background("white")

        su, sv = source_edge
        pts_src = np.array([pos_src[su], pos_src[sv]])
        mesh_src = pv.PolyData(pts_src)
        mesh_src.lines = np.array([2, 0, 1])
        detail.add_mesh(mesh_src.tube(radius=tube_radius), color="#E63946", label=f"{source_key} ({su},{sv})")

        for nid in (su, sv):
            detail.add_point_labels(
                pv.PolyData(pos_src[nid].reshape(1, -1)),
                [str(nid)],
                font_size=14, point_size=8, text_color="red",
            )

        for te in target_edges:
            tu, tv = te
            if tu not in pos_tgt or tv not in pos_tgt:
                continue
            pts_tgt = np.array([pos_tgt[tu], pos_tgt[tv]])
            mesh_tgt = pv.PolyData(pts_tgt)
            mesh_tgt.lines = np.array([2, 0, 1])
            detail.add_mesh(mesh_tgt.tube(radius=tube_radius), color="#457B9D", label=f"{target_key} ({tu},{tv})")

            for nid in (tu, tv):
                detail.add_point_labels(
                    pv.PolyData(pos_tgt[nid].reshape(1, -1)),
                    [str(nid)],
                    font_size=14, point_size=8, text_color="blue",
                )

        detail.add_legend()
        detail.enable_anti_aliasing()
        detail.show()

    def on_pick(point):
        if point is None:
            return
        picked = np.array(point)
        dist, idx = edge_tree.query(picked)
        info = sample_edge_info[idx]
        u, v = info["edge"]
        graph_key = info["graph_key"]
        graph = info["graph"]
        category = info["category"]

        print(f"\n{'=' * 60}")
        print(f"  Category:  {category}")
        print(f"  Graph:     {graph_key}")
        print(f"  Edge:      ({u}, {v})")
        print(f"  Node {u}:   {get_node_coords(graph, u)}")
        print(f"  Node {v}:   {get_node_coords(graph, v)}")
        print(f"  Click dist: {dist:.2f}")

        key = (graph_key, (u, v))
        key_rev = (graph_key, (v, u))
        matched_set = None
        if key in match_info:
            direction, matched_set = match_info[key]
            print(f"  Match:     {direction} -> {matched_set}")
        elif key_rev in match_info:
            direction, matched_set = match_info[key_rev]
            print(f"  Match:     {direction} -> {matched_set}")
        else:
            print("  Match:     (none)")
        print(f"{'=' * 60}")

        if matched_set:
            if graph_key == "g1":
                _show_match_detail(g1, (u, v), "g1", g2, matched_set, "g2", pos1, pos2)
            else:
                _show_match_detail(g2, (u, v), "g2", g1, matched_set, "g1", pos2, pos1)

    # Render the tubes
    plotter = pv.Plotter()
    plotter.set_background("white")

    render_groups = [
        (g1, matched_edges_g1,   pos1, COLORS["matched_g1"],   "G1 Matched"),
        (g2, matched_edges_g2,   pos2, COLORS["matched_g2"],   "G2 Matched"),
        (g1, unmatched_edges_g1, pos1, COLORS["unmatched_g1"], "G1 Unmatched"),
        (g2, unmatched_edges_g2, pos2, COLORS["unmatched_g2"], "G2 Unmatched"),
    ]

    for graph, edges, pos, color, label in render_groups:
        points, lines = _collect_edge_geometry(graph, edges, pos)
        _add_edge_tubes(plotter, points, lines, color, label, tube_radius)

    plotter.enable_point_picking(
        callback=on_pick,
        show_message="Left-click an edge to identify it",
        show_point=True,
        point_size=10,
        color="yellow",
        left_clicking=True,
    )
    plotter.add_legend()
    plotter.enable_anti_aliasing()
    plotter.show(auto_close=False, interactive_update=True)


def viz_final_matches(
    g1: nx.Graph,
    g2: nx.Graph,
    final_matches: list,
    tube_radius: float = 0.8,
):
    """
    Visualize the result of final_matches consolidation.

    Left-click an edge to print its info and open a detail window showing its match pair.
    """
    matched_edges_g1 = set()
    matched_edges_g2 = set()
    g1_to_g2 = {}
    g2_to_g1 = {}

    for g1_edges, g2_edges in final_matches:
        matched_edges_g1.update(g1_edges)
        matched_edges_g2.update(g2_edges)
        for e1 in g1_edges:
            g1_to_g2[e1] = set(g2_edges)
        for e2 in g2_edges:
            g2_to_g1[e2] = set(g1_edges)

    unmatched_edges_g1 = {e for e in g1.edges() if not _is_matched(e, matched_edges_g1)}
    unmatched_edges_g2 = {e for e in g2.edges() if not _is_matched(e, matched_edges_g2)}

    viz_matched_unmatched_interactive(
        g1, g2,
        matched_edges_g1, matched_edges_g2,
        unmatched_edges_g1, unmatched_edges_g2,
        matched_es1_dict=g1_to_g2,
        matched_es2_dict=g2_to_g1,
        tube_radius=tube_radius,
    )

graph_gt = r"E:\sr_data\532\cuff_analysis\graph_comparison_cuff_base\original\vesselvio\Graphs\R_20190216163532_AngelosHyperamia_base_1_RSOM50_wl1_corr_v_rgb_pred.pkl"
graph_i25 = r"E:\sr_data\532\cuff_analysis\graph_comparison_cuff_base\i25\vesselvio\Graphs\R_20190216163532_AngelosHyperamia_base_1_RSOM50_wl1_corr_v_rgb_pred.pkl"
graph_i16 = r"e:\sr_data\532\cuff_analysis\graph_comparison_cuff_base\i16\vesselvio\R_20190216163532_AngelosHyperamia_base_1_RSOM50_wl1_corr_v_rgb_pred.pkl"
graph_i9 = r"e:\sr_data\532\cuff_analysis\graph_comparison_cuff_base\i9\vesselvio\R_20190216163532_AngelosHyperamia_base_1_RSOM50_wl1_corr_v_rgb_pred.pkl"


g_gt = load_graph(graph_gt)
comparison_graphs = {
    "i9":  load_graph(graph_i9),
    "i16": load_graph(graph_i16),
    "i25": load_graph(graph_i25),
}

distance = 13
filter_length = 0.250

all_stats, all_matches = compare_representations(g_gt, comparison_graphs, distance=distance, filter_length=filter_length)

# Visualize a specific comparison interactively (change label as needed)
viz_label = "i9"
g2_viz, fm_viz = all_matches[viz_label]
print_match_summary(g_gt, g2_viz, fm_viz, filter_length=filter_length)
viz_final_matches(g_gt, g2_viz, fm_viz)
print(1)