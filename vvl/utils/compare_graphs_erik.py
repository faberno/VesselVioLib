import os
import pickle
import igraph as ig
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import networkx as nx
import networkx as nx
import plotly.graph_objects as go
import pyvista as pv
import numpy as np
import ast


def load_graph(input_path):
    with open(input_path, "rb") as f:
        g = pickle.load(f)
        g = nx.Graph(g.to_networkx())
    return g


def save_graph(g, output_path):
    # Remove attributes that save_graph deletes
    attrs_to_remove_nodes = ["v_coords"]
    attrs_to_remove_edges = ["radii_list", "coords_list", "original_edge_positions", "original_edge_paths"]

    for node in g.nodes():
        for attr in attrs_to_remove_nodes:
            if attr in g.nodes[node]:
                del g.nodes[node][attr]

    for u, v in g.edges():
        for attr in attrs_to_remove_edges:
            if attr in g.edges[u, v]:
                del g.edges[u, v][attr]

    nx.write_graphml(g, output_path)


def viz_graph(G):

    # Use actual node coordinates
    pos = {n: np.array([float(G.nodes[n]["X"]), float(G.nodes[n]["Y"]), float(G.nodes[n]["Z"])]) for n in G.nodes()}

    # Separate g1 and g2 edges for different colors
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

    # Create edge meshes with tubes for smoother look
    plotter = pv.Plotter()
    plotter.set_background("white")

    if g1_points:
        g1_mesh = pv.PolyData(np.array(g1_points))
        g1_mesh.lines = np.array(g1_lines)
        g1_tubes = g1_mesh.tube(radius=0.8)  # Adjust radius as needed
        plotter.add_mesh(g1_tubes, color="#E63946", label="Graph 1")  # Red

    if g2_points:
        g2_mesh = pv.PolyData(np.array(g2_points))
        g2_mesh.lines = np.array(g2_lines)
        g2_tubes = g2_mesh.tube(radius=0.8)
        plotter.add_mesh(g2_tubes, color="#457B9D", label="Graph 2")  # Blue

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
    """
    Visualize matched and unmatched edges from two graphs with distinct colors.

    Parameters
    ----------
    g1 : nx.Graph
        First graph
    g2 : nx.Graph
        Second graph
    matched_edges_g1 : set
        Edge tuples from g1 that were matched
    matched_edges_g2 : set
        Edge tuples from g2 that were matched
    unmatched_edges_g1 : set
        Edge tuples from g1 that were not matched
    unmatched_edges_g2 : set
        Edge tuples from g2 that were not matched
    tube_radius : float
        Radius of the tube visualization for edges
    """

    def get_pos(graph):
        return {
            n: np.array([float(graph.nodes[n]["X"]), float(graph.nodes[n]["Y"]), float(graph.nodes[n]["Z"])])
            for n in graph.nodes()
        }

    def collect_edge_geometry(graph, edges, pos):
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

    pos1 = get_pos(g1)
    pos2 = get_pos(g2)

    # Collect geometry for each group
    matched_g1_pts, matched_g1_lines = collect_edge_geometry(g1, matched_edges_g1, pos1)
    matched_g2_pts, matched_g2_lines = collect_edge_geometry(g2, matched_edges_g2, pos2)
    unmatched_g1_pts, unmatched_g1_lines = collect_edge_geometry(g1, unmatched_edges_g1, pos1)
    unmatched_g2_pts, unmatched_g2_lines = collect_edge_geometry(g2, unmatched_edges_g2, pos2)

    # Define colors for each group
    colors = {
        "matched_g1": "#2E8B57",  # Sea green
        "matched_g2": "#4169E1",  # Royal blue
        "unmatched_g1": "#DC143C",  # Crimson
        "unmatched_g2": "#FF8C00",  # Dark orange
    }

    plotter = pv.Plotter()
    plotter.set_background("white")

    groups = [
        (matched_g1_pts, matched_g1_lines, colors["matched_g1"], "G1 Matched"),
        (matched_g2_pts, matched_g2_lines, colors["matched_g2"], "G2 Matched"),
        (unmatched_g1_pts, unmatched_g1_lines, colors["unmatched_g1"], "G1 Unmatched"),
        (unmatched_g2_pts, unmatched_g2_lines, colors["unmatched_g2"], "G2 Unmatched"),
    ]

    for points, lines, color, label in groups:
        if points:
            mesh = pv.PolyData(np.array(points))
            mesh.lines = np.array(lines)
            tubes = mesh.tube(radius=tube_radius)
            plotter.add_mesh(tubes, color=color, label=label)

    plotter.add_legend()
    plotter.enable_anti_aliasing()
    plotter.show(auto_close=False, interactive_update=True)


def get_node_coords(graph: nx.Graph, node_id: str) -> np.ndarray:
    """Extract XYZ coordinates from a node."""
    d = graph.nodes[node_id]
    return np.array([float(d["X"]), float(d["Y"]), float(d["Z"])])


def get_all_node_coords(graph: nx.Graph) -> tuple[list, np.ndarray]:
    """Get all node IDs and their coordinates as arrays."""
    node_ids = list(graph.nodes())
    coords = np.array([get_node_coords(graph, n) for n in node_ids])
    return node_ids, coords


def get_edge_direction(G, edge):  # -> NDArray[Any] | NDArray[float64]:# -> NDArray[Any] | NDArray[float64]:
    """Get the 3D direction vector of an edge."""
    u, v = edge
    pos_u = np.array([G.nodes[u]["X"], G.nodes[u]["Y"], G.nodes[u]["Z"]])
    pos_v = np.array([G.nodes[v]["X"], G.nodes[v]["Y"], G.nodes[v]["Z"]])
    direction = pos_v - pos_u
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


def angular_similarity(G1, edge1, G2, edge2):  # -> Any:
    """
    Compute angular similarity between two edges.
    Returns angle in degrees (0 = parallel, 90 = perpendicular).
    Ignores direction (treats edges as undirected).
    """
    dir1 = get_edge_direction(G1, edge1)
    dir2 = get_edge_direction(G2, edge2)

    # Use absolute value of dot product to ignore direction
    cos_angle = np.abs(np.clip(np.dot(dir1, dir2), -1, 1))
    angle_deg = np.degrees(np.arccos(cos_angle))

    return angle_deg


def get_edge_points(G, edge):
    """
    Get points along an edge.
    If edge has 'path' attribute with intermediate points, use those.
    Otherwise, interpolate between endpoints.
    """
    u, v = edge
    edge_attr = G.edges[edge]
    coords = edge_attr["coords_list"]
    if isinstance(coords, str):
        # Parse if stored as string
        import ast

        coords = ast.literal_eval(coords)
    coords = np.array(coords)
    return coords


# def check_proximity(G1, edge1, G2, edge2, max_distance=5.0):
#     """
#     Check if all points on the shorter edge are within max_distance
#     of at least one point on the longer edge.

#     Returns:
#         is_similar: bool - True if all points satisfy the criterion
#         max_min_dist: float - the maximum of the minimum distances
#         distances: array - minimum distance for each point on shorter edge
#     """
#     points1 = get_edge_points(G1, edge1)
#     points2 = get_edge_points(G2, edge2)

#     # Determine shorter and longer
#     len1 = np.linalg.norm(points1[-1] - points1[0])
#     len2 = np.linalg.norm(points2[-1] - points2[0])

#     if len1 <= len2:
#         shorter, longer = points1, points2
#     else:
#         shorter, longer = points2, points1

#     # For each point on shorter, find min distance to any point on longer
#     min_distances = np.zeros(len(shorter))
#     for i, pt in enumerate(shorter):
#         dists = np.linalg.norm(longer - pt, axis=1)
#         min_distances[i] = np.min(dists)

#     max_min_dist = np.max(min_distances)
#     is_similar = max_min_dist <= max_distance

#     return is_similar, max_min_dist, min_distances, len1, len2


# def compare_edges(G1, edge1, G2, edge2, angle_threshold=15.0, distance_threshold=5.0, edge_case_small_len=10.0):
#     """
#     Full comparison of two edges.

#     Returns dict with:
#         - angle_deg: angle between edges in degrees
#         - angles_similar: True if angle <= threshold
#         - proximity_satisfied: True if all points within distance
#         - max_min_distance: worst-case minimum distance
#         - is_similar: True if both criteria met
#     """
#     angle = angular_similarity(G1, edge1, G2, edge2)
#     prox_ok, max_min_dist, min_dists, len_e1, len_e2 = check_proximity(G1, edge1, G2, edge2, distance_threshold)

#     is_similar = 1.0 if (angle <= angle_threshold) and prox_ok else 0.0
#     # If proximity is good then edge_similarity may be okay as long as one of the vessels is extremely small.
#     if is_similar == 0.0 and prox_ok and (len_e1 < edge_case_small_len or len_e2 < edge_case_small_len):
#         is_similar = 0.5
#     return {
#         "angle_deg": angle,
#         "angles_similar": angle <= angle_threshold,
#         "proximity_satisfied": prox_ok,
#         "max_min_distance": max_min_dist,
#         "min_dists": min_dists,
#         "edge_lengths": (len_e1, len_e2),
#         "is_similar": is_similar,
#     }


def find_nearby_nodes(edge_coords: np.ndarray, graph: nx.Graph, max_distance: float) -> set:
    """
    Find all nodes in graph that are within max_distance of any point in edge_coords.

    Parameters
    ----------
    edge_coords : np.ndarray
        Array of shape (N, 3) with coordinates along the edge
    graph : nx.Graph
        Graph to search for nearby nodes
    max_distance : float
        Maximum distance threshold

    Returns
    -------
    set
        Node IDs that are within max_distance of any edge coordinate
    """
    if len(edge_coords) == 0:
        return set()

    # Build KDTree from edge coordinates for efficient distance queries
    edge_tree = KDTree(edge_coords)

    nearby_nodes = set()
    for node in graph.nodes():
        node_coord = get_node_coords(graph, node)
        # Find distance to closest point on the edge
        dist, _ = edge_tree.query(node_coord)
        if dist <= max_distance:
            nearby_nodes.add(node)

    return nearby_nodes


def build_candidate_subgraph(graph: nx.Graph, candidate_nodes: set) -> nx.Graph:
    """
    Build a subgraph containing only edges where BOTH endpoints are in candidate_nodes.

    Parameters
    ----------
    graph : nx.Graph
        Original graph
    candidate_nodes : set
        Set of node IDs to include

    Returns
    -------
    nx.Graph
        Subgraph with only edges between candidate nodes
    """
    subgraph = nx.Graph()

    for node in candidate_nodes:
        if node in graph.nodes:
            subgraph.add_node(node, **graph.nodes[node])

    for u, v in graph.edges():
        if u in candidate_nodes and v in candidate_nodes:
            subgraph.add_edge(u, v, **graph.edges[u, v])

    return subgraph


def get_edge_length(g, u, v):
    """Get physical length of an edge."""
    edge_data = g.edges[u, v]
    if "coords_list" in edge_data:
        coords = edge_data["coords_list"]
        if isinstance(coords, str):
            coords = ast.literal_eval(coords)
        coords = np.array(coords)
        if len(coords) >= 2:
            # Sum of segment lengths
            return np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    # Fallback: distance between endpoints
    p1 = get_node_coords(g, u)
    p2 = get_node_coords(g, v)
    return np.linalg.norm(p2 - p1)


def path_length(path, subgraph):
    """Total physical length of a path."""
    total = 0
    for i in range(len(path) - 1):
        total += get_edge_length(subgraph, path[i], path[i + 1])
    return total


# def find_angular_path_in_subgraph(subgraph: nx.Graph, edge, source_graph: nx.Graph):
#     target_angle = get_edge_direction(source_graph, edge)
#     start_coord = get_node_coords(source_graph, edge[0])
#     end_coord = get_node_coords(source_graph, edge[1])

#     start_dists = []
#     for i, node in enumerate(subgraph.nodes):
#         coord = get_node_coords(subgraph, node)
#         start_dists.append((i, np.linalg.norm(coord - start_coord)))

#     start_idx = min(start_dists, key=lambda x: x[1])[0]

#     start_node = subgraph.nodes[start_idx]

#     for i, node in enumerate(subgraph.neighbors(start_node)):
#         pass


def find_longest_path_in_subgraph(subgraph: nx.Graph) -> list:
    """
    Find the longest path in a subgraph using DFS from endpoints.

    Parameters
    ----------
    subgraph : nx.Graph
        Graph to find longest path in

    Returns
    -------
    list
        List of node IDs forming the longest path
    """
    if len(subgraph.nodes) == 0:
        return []

    if len(subgraph.nodes) == 1:
        return list(subgraph.nodes)

    # Find endpoints (degree 1 nodes)
    endpoints = [n for n in subgraph.nodes if subgraph.degree(n) == 1]

    # If no endpoints (cycle), pick arbitrary start
    if not endpoints:
        endpoints = [list(subgraph.nodes)[0]]

    # Find longest path using DFS from each endpoint
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


def find_length_matched_path_in_subgraph(subgraph: nx.Graph, start: int, end, source_len) -> list:
    """
    Find the path with the closest matching length in a subgraph using DFS from endpoints.

    Parameters
    ----------
    subgraph : nx.Graph
        Graph to find longest path in
    start: int
        Node at which to start
    source_len: float
        Target length of path
    Returns
    -------
    list
        List of node IDs forming the longest path
    """
    if len(subgraph.nodes) == 0:
        return []

    if len(subgraph.nodes) == 1:
        return list(subgraph.nodes)

    best_p = []
    best_len = np.inf

    # First, try to see if there is a good match for target node. If yes, try to find a way there, if no or if fails then just try to match path length
    end_node, end_dist = end
    if end_dist < 5:
        try:
            best_p = nx.shortest_path(subgraph, source=start, target=end_node, weight="length")  # Will return empty if start==end
        except nx.exception.NetworkXNoPath as e:
            pass
    if len(best_p) < 2:
        # Find best path-length using DFS from each endpoint
        stack = [(start, [start], {start})]
        while stack:
            node, path, visited = stack.pop()
            neighbors = [n for n in subgraph.neighbors(node) if n not in visited]

            plen = path_length(path, subgraph)
            if abs(plen - source_len) < best_len:
                best_p = path
                best_len = abs(plen - source_len)

            for neighbor in neighbors:
                stack.append((neighbor, path + [neighbor], visited | {neighbor}))

    return best_p


def crop_path_to_endpoints(
    path: list,
    graph: nx.Graph,
    start_coord: np.ndarray,
    end_coord: np.ndarray,
) -> list:
    """
    Crop a path so that its start and end nodes are closest to the given coordinates.

    Parameters
    ----------
    path : list
        List of node IDs forming the path
    graph : nx.Graph
        Graph containing node coordinates
    start_coord : np.ndarray
        Target coordinate for the start of the cropped path
    end_coord : np.ndarray
        Target coordinate for the end of the cropped path

    Returns
    -------
    list
        Cropped path where start is closest to start_coord and end is closest to end_coord
    """
    if len(path) <= 1:
        return path

    # Find the node closest to start_coord
    start_dists = []
    for i, node in enumerate(path):
        coord = get_node_coords(graph, node)
        start_dists.append((i, np.linalg.norm(coord - start_coord)))

    # Find the node closest to end_coord
    end_dists = []
    for i, node in enumerate(path):
        coord = get_node_coords(graph, node)
        end_dists.append((i, np.linalg.norm(coord - end_coord)))

    start_idx = min(start_dists, key=lambda x: x[1])[0]
    end_idx = min(end_dists, key=lambda x: x[1])[0]

    # Ensure start_idx < end_idx (swap if needed)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Crop the path
    return path[start_idx : end_idx + 1]


def get_matched_edges_for_edge(
    edge: tuple,
    g1: nx.Graph,
    g2: nx.Graph,
    max_distance: float = 5.0,
) -> set:
    """
    Find matching edges in g2 for a given edge in g1 using node-based matching.

    Algorithm:
    1. Get coords_list of the edge in g1
    2. Find all nodes in g2 within max_distance of any point in coords_list
    3. Build a subgraph of g2 with only those candidate nodes
    4. Find the longest path in that subgraph
    5. Crop the path so start/end are closest to original edge endpoints

    Parameters
    ----------
    edge : tuple
        Edge (u, v) from g1
    g1 : nx.Graph
        Source graph
    g2 : nx.Graph
        Target graph to find matches in
    max_distance : float
        Maximum distance for node matching

    Returns
    -------
    set
        Set of edge tuples from g2 that form the matched path
    """
    u, v = edge
    source_len = get_edge_length(g1, u, v)
    start_coord = get_node_coords(g1, u)
    end_coord = get_node_coords(g1, v)
    edge_coords = get_edge_points(g1, edge)
    # Find nearby nodes in g2
    candidate_nodes = find_nearby_nodes(edge_coords, g2, max_distance)

    if len(candidate_nodes) < 2:
        return set()

    subgraph = build_candidate_subgraph(g2, candidate_nodes)

    if len(subgraph.edges) == 0:
        return set()

    # Find best matches for start and end node
    start_dists = []
    for i, node in enumerate(candidate_nodes):
        coord = get_node_coords(g2, node)
        start_dists.append((node, np.linalg.norm(coord - start_coord)))
    start_match = min(start_dists, key=lambda x: x[1])
    # Find the node closest to end_coord
    end_dists = []
    for i, node in enumerate(candidate_nodes):
        coord = get_node_coords(g2, node)
        end_dists.append((node, np.linalg.norm(coord - end_coord)))
    end_match = min(end_dists, key=lambda x: x[1])

    if start_match[1] > end_match[1]:
        best_path = find_length_matched_path_in_subgraph(subgraph, end_match[0], start_match, source_len)
    else:
        best_path = find_length_matched_path_in_subgraph(subgraph, start_match[0], end_match, source_len)

    if len(best_path) < 2:
        return set()

    # Crop to match original edge endpoints
    cropped_path = crop_path_to_endpoints(best_path, g2, start_coord, end_coord)

    if len(cropped_path) < 2:
        return set()
    # TODO: Fréchet distance
    if angular_similarity(g1, (u, v), g2, (cropped_path[0], cropped_path[-1])) > 30:
        return set()

    # Convert path to edges
    matched_edges = set()
    for i in range(len(cropped_path) - 1):
        n1, n2 = cropped_path[i], cropped_path[i + 1]
        # Use canonical edge order from g2
        # if g2.has_edge(n1, n2):
        #     matched_edges.add((n1, n2))
        # elif g2.has_edge(n2, n1):
        #     matched_edges.add((n2, n1))
        matched_edges.add((n2, n1))
    return matched_edges


def detect_split_edges(e1, similarites, g):
    """
    Sometimes a edge in one graph will be split into multiple edges in the other. This heuristic tries to account for this.

    :param e1: original edge to compare.
    :param similarites: Dictionary with similarities for each edge in g2 compared
    :param g: graph to compare to.
    """

    lens = [sim["edge_lengths"][1] for sim in similarites.values()]
    if max(lens) > similarites[0]["edge_lengths"][0]:
        raise ValueError("Can only detect split vessels for elements smaller than the origin vessel.")


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
    Left-click picks, drag to rotate still works.
    """

    def get_pos(graph):
        return {
            n: np.array([float(graph.nodes[n]["X"]), float(graph.nodes[n]["Y"]), float(graph.nodes[n]["Z"])])
            for n in graph.nodes()
        }

    def collect_edge_geometry(graph, edges, pos):
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

    pos1 = get_pos(g1)
    pos2 = get_pos(g2)

    # Build lookup: sample start, mid, end of every edge for KDTree nearest-neighbour search
    sample_points = []
    sample_edge_info = []

    edge_groups = [
        (g1, matched_edges_g1, pos1, "g1", "G1 Matched"),
        (g2, matched_edges_g2, pos2, "g2", "G2 Matched"),
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

        # Source edge
        su, sv = source_edge
        pts_src = np.array([pos_src[su], pos_src[sv]])
        mesh_src = pv.PolyData(pts_src)
        mesh_src.lines = np.array([2, 0, 1])
        detail.add_mesh(mesh_src.tube(radius=tube_radius), color="#E63946", label=f"{source_key} ({su},{sv})")

        # Source node labels
        for nid in (su, sv):
            detail.add_point_labels(
                pv.PolyData(pos_src[nid].reshape(1, -1)),
                [str(nid)],
                font_size=14,
                point_size=8,
                text_color="red",
            )

        # Target matched edges
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
                    font_size=14,
                    point_size=8,
                    text_color="blue",
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
            print(f"  Match:     (none)")
        print(f"{'=' * 60}")

        # Open detail window if there is a match
        if matched_set:
            if graph_key == "g1":
                _show_match_detail(g1, (u, v), "g1", g2, matched_set, "g2", pos1, pos2)
            else:
                _show_match_detail(g2, (u, v), "g2", g1, matched_set, "g1", pos2, pos1)

    # ---- render the same tubes as viz_matched_unmatched ----
    matched_g1_pts, matched_g1_lines = collect_edge_geometry(g1, matched_edges_g1, pos1)
    matched_g2_pts, matched_g2_lines = collect_edge_geometry(g2, matched_edges_g2, pos2)
    unmatched_g1_pts, unmatched_g1_lines = collect_edge_geometry(g1, unmatched_edges_g1, pos1)
    unmatched_g2_pts, unmatched_g2_lines = collect_edge_geometry(g2, unmatched_edges_g2, pos2)

    colors = {
        "matched_g1": "#2E8B57",
        "matched_g2": "#4169E1",
        "unmatched_g1": "#DC143C",
        "unmatched_g2": "#FF8C00",
    }

    plotter = pv.Plotter()
    plotter.set_background("white")

    groups = [
        (matched_g1_pts, matched_g1_lines, colors["matched_g1"], "G1 Matched"),
        (matched_g2_pts, matched_g2_lines, colors["matched_g2"], "G2 Matched"),
        (unmatched_g1_pts, unmatched_g1_lines, colors["unmatched_g1"], "G1 Unmatched"),
        (unmatched_g2_pts, unmatched_g2_lines, colors["unmatched_g2"], "G2 Unmatched"),
    ]

    for points, lines, color, label in groups:
        if points:
            mesh = pv.PolyData(np.array(points))
            mesh.lines = np.array(lines)
            tubes = mesh.tube(radius=tube_radius)
            plotter.add_mesh(tubes, color=color, label=label, pickable=True)

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

    Colors
    ------
    Sea green   : matched g1 edges
    Royal blue  : matched g2 edges
    Crimson     : unmatched g1 edges
    Dark orange : unmatched g2 edges

    Left-click an edge to print its info and open a detail window showing its match pair.

    Parameters
    ----------
    g1 : nx.Graph
        First graph (e.g. ground truth).
    g2 : nx.Graph
        Second graph (e.g. reconstruction).
    final_matches : list of (g1_edges_set, g2_edges_set)
        Each element is a pair of sets: the g1 edges and the corresponding g2 edges.
    tube_radius : float
        Radius of tube visualization.
    """
    matched_edges_g1 = set()
    matched_edges_g2 = set()
    g1_to_g2 = {}  # g1_edge -> set of g2 edges
    g2_to_g1 = {}  # g2_edge -> set of g1 edges

    for g1_edges, g2_edges in final_matches:
        matched_edges_g1.update(g1_edges)
        matched_edges_g2.update(g2_edges)
        for e1 in g1_edges:
            g1_to_g2[e1] = set(g2_edges)
        for e2 in g2_edges:
            g2_to_g1[e2] = set(g1_edges)

    def is_matched(edge, matched_set):
        return edge in matched_set or (edge[1], edge[0]) in matched_set

    unmatched_edges_g1 = {e for e in g1.edges() if not is_matched(e, matched_edges_g1)}
    unmatched_edges_g2 = {e for e in g2.edges() if not is_matched(e, matched_edges_g2)}

    viz_matched_unmatched_interactive(
        g1,
        g2,
        {},#matched_edges_g1,
        {},#matched_edges_g2,
        unmatched_edges_g1,
        unmatched_edges_g2,
        matched_es1_dict=g1_to_g2,
        matched_es2_dict=g2_to_g1,
        tube_radius=tube_radius,
    )


def print_match_summary(g1: nx.Graph, g2: nx.Graph, final_matches: list):
    """
    Print per-match and aggregate errors for length, volume, and radius_avg.

    For each match (g1_edges, g2_edges) the g1 and g2 sides are each
    aggregated as:
      - length      : sum of edge lengths
      - volume      : sum of edge volumes
      - radius_avg  : length-weighted mean radius

    Errors are g2 - g1 (absolute) and (g2 - g1) / g1 * 100 (relative %).
    Aggregate statistics (mean, median, std) are printed at the end.
    """

    def _agg(graph, edges):
        lengths = []
        volumes = []
        radii = []
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

    rows = []
    for i, (g1_edges, g2_edges) in enumerate(final_matches):
        l1, v1, r1 = _agg(g1, g1_edges)
        l2, v2, r2 = _agg(g2, g2_edges)
        rows.append(
            {
                "i": i,
                "n_g1": len(g1_edges),
                "n_g2": len(g2_edges),
                "len_g1": l1,
                "len_g2": l2,
                "len_abs": l2 - l1,
                "len_rel": (l2 - l1) / l1 * 100 if l1 > 0 else float("nan"),
                "vol_g1": v1,
                "vol_g2": v2,
                "vol_abs": v2 - v1,
                "vol_rel": (v2 - v1) / v1 * 100 if v1 > 0 else float("nan"),
                "rad_g1": r1,
                "rad_g2": r2,
                "rad_abs": r2 - r1,
                "rad_rel": (r2 - r1) / r1 * 100 if r1 > 0 else float("nan"),
            }
        )

    if not rows:
        print("No matches to summarize.")
        return

    # ---- per-match table ----
    rows.sort(key=lambda r: r["len_g1"])
    header = f"{'#':>5}  {'g1e':>4}{'g2e':>4}  {'len_g1':>8}{'len_g2':>8}{'len_err':>9}{'len_%':>7}  {'vol_g1':>10}{'vol_g2':>10}{'vol_err':>11}{'vol_%':>7}  {'rad_g1':>7}{'rad_g2':>7}{'rad_err':>8}{'rad_%':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['i']:>5}  {r['n_g1']:>4}{r['n_g2']:>4}"
            f"  {r['len_g1']:>8.1f}{r['len_g2']:>8.1f}{r['len_abs']:>+9.1f}{r['len_rel']:>+7.1f}"
            f"  {r['vol_g1']:>10.1f}{r['vol_g2']:>10.1f}{r['vol_abs']:>+11.1f}{r['vol_rel']:>+7.1f}"
            f"  {r['rad_g1']:>7.2f}{r['rad_g2']:>7.2f}{r['rad_abs']:>+8.3f}{r['rad_rel']:>+7.1f}"
        )

    # ---- aggregate stats ----
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

    # ---- unmatched edge stats ----
    matched_edges_g1 = set()
    matched_edges_g2 = set()
    for g1_edges, g2_edges in final_matches:
        matched_edges_g1.update(g1_edges)
        matched_edges_g2.update(g2_edges)

    def is_matched(edge, matched_set):
        return edge in matched_set or (edge[1], edge[0]) in matched_set

    unmatched_g1 = [e for e in g1.edges() if not is_matched(e, matched_edges_g1)]
    unmatched_g2 = [e for e in g2.edges() if not is_matched(e, matched_edges_g2)]

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
        print(f"  {'-'*60}")
        for name, arr in [("length", lens), ("volume", vols), ("radius_avg", rads)]:
            total = f"{arr.sum():>12.1f}" if name != "radius_avg" else f"{'—':>12}"
            print(f"  {name:16} {arr.mean():>10.3f} {np.median(arr):>10.3f} {arr.std():>10.3f} {total}")

    _print_unmatched_stats("G1 unmatched", g1, unmatched_g1)
    _print_unmatched_stats("G2 unmatched", g2, unmatched_g2)


def run_matching(g1: nx.Graph, g2: nx.Graph, distance: float = 13.0) -> list:
    """
    Run the full bidirectional matching pipeline between g1 and g2.

    Returns
    -------
    list of (g1_edges_set, g2_edges_set) tuples
    """
    matched_es1_dict = {}
    matched_es2_dict = {}

    for e1 in tqdm(g1.edges, desc="Matching g1→g2"):
        matched = get_matched_edges_for_edge(e1, g1, g2, max_distance=distance)
        if matched:
            matched_es1_dict[e1] = matched

    for e2 in tqdm(g2.edges, desc="Matching g2→g1"):
        matched = get_matched_edges_for_edge(e2, g2, g1, max_distance=distance)
        if matched:
            matched_es2_dict[e2] = matched

    """
    TODO: 
    ** Final Matchings **
    The way our vessel matching algorithm works means that all matches from e1 of g1 can only be similarly long as e1. And vice-versa for all e2 of g2. 
    This means that we can't just take one graph_matches but we need a smart way to combine them.

    The problem is that match-finding is directional. Meaning a match will only ever be as long as the source_edge.
    How do we identify good edges:
        Accept all matchings from g1
        Examine all edges of g2 with multi-edge matching in g1: (1:1 matchings can be ignored since they will have been considered by the other match as well)
            - Multi-edge matching likely means good match (NOTE: Can we ensure that this is a good match)
            - Now identify if any g1-> g2 matchings can be removed:
                - if source_edge of g2_multi_match is found as match from g1
                    -> if this is a 1:1 matching: Compute Frechet distance and delete the bad and keep the good (If not rejected in any of the occurences of above)
                    -> If not then:
                        check if g1_source_edge is included in original g2_matching
                        -> This means that g2_source is part of g1_match AND g1_source is part of g2_match 
                            -> remove both and create new final_matching from these two matchings
                        -> If not included in original g2_matching, keep both since it is probably a unique matching.
                - for individual edges along the path of the multi-edge matching from source_e_g2:
                    -> Remove any instances of 1:1 matchings -> Since they will be shorter and thus more likely to be wrong
                    -> Other instances of multi-path matchings -> Don't do anything since it is probably a unique matching

        Examine all edges that are present in multiple matchings
            - If there are multiple 1:1 matchings keep only the one with the best Frechet distance

    """
    final_matches = []
    for e1_key, e1_match in matched_es1_dict.items():
        final_matches.append(({e1_key}, e1_match))
    for e2_key, e2_match in matched_es2_dict.items():
        if len(e2_match) < 2:
            continue
        final_matches.append((e2_match, {e2_key}))

    return final_matches


def get_match_stats(g1: nx.Graph, g2: nx.Graph, final_matches: list) -> dict:
    """
    Compute summary statistics for a set of final_matches.

    Returns a flat dict with keys covering:
      - matching overview (counts, unmatched totals)
      - per-feature error stats (mean/median abs and rel for length, volume, radius_avg)
    """

    def _agg(graph, edges):
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

    # Per-match errors
    len_abs, len_rel, vol_abs, vol_rel, rad_abs, rad_rel = [], [], [], [], [], []
    matched_g1, matched_g2 = set(), set()

    for g1_edges, g2_edges in final_matches:
        matched_g1.update(g1_edges)
        matched_g2.update(g2_edges)
        l1, v1, r1 = _agg(g1, g1_edges)
        l2, v2, r2 = _agg(g2, g2_edges)
        len_abs.append(l2 - l1)
        len_rel.append((l2 - l1) / l1 * 100 if l1 > 0 else float("nan"))
        vol_abs.append(v2 - v1)
        vol_rel.append((v2 - v1) / v1 * 100 if v1 > 0 else float("nan"))
        rad_abs.append(r2 - r1)
        rad_rel.append((r2 - r1) / r1 * 100 if r1 > 0 else float("nan"))

    def _s(arr, fn):
        a = np.array(arr)
        a = a[np.isfinite(a)]
        return fn(a).item() if len(a) > 0 else float("nan")

    def is_matched(edge, matched_set):
        return edge in matched_set or (edge[1], edge[0]) in matched_set

    unmatched_g1 = [e for e in g1.edges() if not is_matched(e, matched_g1)]
    unmatched_g2 = [e for e in g2.edges() if not is_matched(e, matched_g2)]

    def _totals(graph, edges):
        total_len = sum(float(graph.edges[u if graph.has_edge(u, v) else v, v if graph.has_edge(u, v) else u].get("length", 0.0)) for u, v in edges)
        mean_len = np.mean([float(graph.edges[u if graph.has_edge(u, v) else v, v if graph.has_edge(u, v) else u].get("length", 0.0)) for u, v in edges])
        std_len = np.std([float(graph.edges[u if graph.has_edge(u, v) else v, v if graph.has_edge(u, v) else u].get("length", 0.0)) for u, v in edges])
        total_vol = sum(float(graph.edges[u if graph.has_edge(u, v) else v, v if graph.has_edge(u, v) else u].get("volume", 0.0)) for u, v in edges)
        mean_vol = np.mean([float(graph.edges[u if graph.has_edge(u, v) else v, v if graph.has_edge(u, v) else u].get("volume", 0.0)) for u, v in edges])
        std_vol = np.std([float(graph.edges[u if graph.has_edge(u, v) else v, v if graph.has_edge(u, v) else u].get("volume", 0.0)) for u, v in edges])

        return total_len, mean_len, std_len, total_vol, mean_vol, std_vol


    unm_total_len_g1, unm_mean_len_g1, unm_std_len_g1, unm_total_vol_g1, unm_mean_vol_g1, unm_std_vol_g1 = _totals(g1, unmatched_g1)
    unm_total_len_g2, unm_mean_len_g2, unm_std_len_g2, unm_total_vol_g2, unm_mean_vol_g2, unm_std_vol_g2 = _totals(g2, unmatched_g2)

    return {
        "n_matches":          len(final_matches),
        "n_matched_g1":       len(matched_g1),
        "n_matched_g2":       len(matched_g2),
        "n_unmatched_g1":     len(unmatched_g1),
        "n_unmatched_g2":     len(unmatched_g2),
        "unm_len_total_g1":         unm_total_len_g1,
        "unm_len_total_g2":         unm_total_len_g2,
        "unm_len_mean_g1":         unm_mean_len_g1,
        "unm_len_mean_g2":         unm_mean_len_g2,
        "unm_len_std_g1":         unm_std_len_g1,
        "unm_len_std_g2":         unm_std_len_g2,
        "unm_vol_total_g1":         unm_total_vol_g1,
        "unm_vol_total_g2":         unm_total_vol_g2,
        "unm_vol_mean_g1":         unm_mean_vol_g1,
        "unm_vol_mean_g2":         unm_mean_vol_g2,
        "unm_vol_std_g1":         unm_std_vol_g1,
        "unm_vol_std_g2":         unm_std_vol_g2,
        "mean_len_abs":       _s(len_abs, np.mean),
        "median_len_abs":     _s(len_abs, np.median),
        "mean_len_rel":       _s(len_rel, np.mean),
        "median_len_rel":     _s(len_rel, np.median),
        "mean_vol_abs":       _s(vol_abs, np.mean),
        "median_vol_abs":     _s(vol_abs, np.median),
        "mean_vol_rel":       _s(vol_rel, np.mean),
        "median_vol_rel":     _s(vol_rel, np.median),
        "mean_rad_abs":       _s(rad_abs, np.mean),
        "median_rad_abs":     _s(rad_abs, np.median),
        "mean_rad_rel":       _s(rad_rel, np.mean),
        "median_rad_rel":     _s(rad_rel, np.median),
    }


def compare_representations(
    g_gt: nx.Graph,
    comparison_graphs: dict,
    distance: float = 13.0,
):
    """
    Compare g_gt against multiple reconstructions and print a side-by-side summary table.

    Parameters
    ----------
    g_gt : nx.Graph
        Ground-truth graph.
    comparison_graphs : dict
        Ordered mapping of label -> nx.Graph, e.g. {"i9": g_i9, "i16": g_i16, "i25": g_i25}.
    distance : float
        Max matching distance (voxels).
    """
    labels = list(comparison_graphs.keys())
    all_stats = {}
    all_matches = {}

    for label, g2 in comparison_graphs.items():
        print(f"\n{'='*60}")
        print(f"  Matching GT vs {label}")
        print(f"{'='*60}")
        fm = run_matching(g_gt, g2, distance=distance)
        all_matches[label] = (g2, fm)
        all_stats[label] = get_match_stats(g_gt, g2, fm)

    # ---- side-by-side table ----
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

    print(f"\n{'--- Matching overview':}")
    print(_row("Matched pairs",          "n_matches",      integer=True))
    print(_row("Matched G1 edges",       "n_matched_g1",   integer=True))
    print(_row("Matched G2 edges",       "n_matched_g2",   integer=True))
    print(_row("Unmatched G1 edges",     "n_unmatched_g1", integer=True))
    print(_row("Unmatched G2 edges",     "n_unmatched_g2", integer=True))
    print(_row("Unmatched G1 total length",    "unm_len_total_g1",     fmt="{:.1f}"))
    print(_row("Unmatched G2 total length",    "unm_len_total_g2",     fmt="{:.1f}"))
    print(_row("Unmatched G1 mean length",    "unm_len_mean_g1",     fmt="{:.1f}"))
    print(_row("Unmatched G2 mean length",    "unm_len_mean_g2",     fmt="{:.1f}"))
    print(_row("Unmatched G1 std length",    "unm_len_std_g1",     fmt="{:.1f}"))
    print(_row("Unmatched G2 std length",    "unm_len_std_g2",     fmt="{:.1f}"))
    print(_row("Unmatched G1 total volume",    "unm_vol_total_g1",     fmt="{:.7f}"))
    print(_row("Unmatched G2 total volume",    "unm_vol_total_g2",     fmt="{:.7f}"))
    print(_row("Unmatched G1 mean volume",    "unm_vol_mean_g1",     fmt="{:.7f}"))
    print(_row("Unmatched G2 mean volume",    "unm_vol_mean_g2",     fmt="{:.7f}"))
    print(_row("Unmatched G1 std volume",    "unm_vol_std_g1",     fmt="{:.7f}"))
    print(_row("Unmatched G2 std volume",    "unm_vol_std_g2",     fmt="{:.7f}"))

    print(f"\n{'--- Length error (vx)':}")
    print(_row("  Mean abs",    "mean_len_abs",   fmt="{:>+.2f}"))
    print(_row("  Median abs",  "median_len_abs", fmt="{:>+.2f}"))
    print(_row("  Mean rel %",  "mean_len_rel",   fmt="{:>+.1f}"))
    print(_row("  Median rel %","median_len_rel", fmt="{:>+.1f}"))

    print(f"\n{'--- Volume error':}")
    print(_row("  Mean abs",    "mean_vol_abs",   fmt="{:>+.2f}"))
    print(_row("  Median abs",  "median_vol_abs", fmt="{:>+.2f}"))
    print(_row("  Mean rel %",  "mean_vol_rel",   fmt="{:>+.1f}"))
    print(_row("  Median rel %","median_vol_rel", fmt="{:>+.1f}"))

    print(f"\n{'--- Radius error (vx)':}")
    print(_row("  Mean abs",    "mean_rad_abs",   fmt="{:>+.3f}"))
    print(_row("  Median abs",  "median_rad_abs", fmt="{:>+.3f}"))
    print(_row("  Mean rel %",  "mean_rad_rel",   fmt="{:>+.1f}"))
    print(_row("  Median rel %","median_rad_rel", fmt="{:>+.1f}"))

    print()
    return all_stats, all_matches


def get_node_dists_from_edge(e1, g1, g2, edge_match):
    start_node = e1[0]
    start_coord = get_node_coords(g1, start_node)
    end_node = e1[1]
    end_coord = get_node_coords(g1, end_node)
    start_dists = []
    end_dists = []

    for node in {num for sublist in edge_match for num in sublist}:
        coord = get_node_coords(g2, node)
        start_dists.append(np.linalg.norm(coord - start_coord))
        end_dists.append(np.linalg.norm(coord - end_coord))
    return min(start_dists) + min(end_dists)


def create_unmatched_edges_graph(
    graph1: nx.Graph,
    graph2: nx.Graph,
    unmatched_edges_g1: set,
    unmatched_edges_g2: set,
) -> nx.Graph:
    """
    Create a graph containing only unmatched edges and their endpoint nodes.

    This function extracts all edges that could not be matched between the two
    input graphs and combines them into a single surplus graph. Only the nodes
    that are endpoints of unmatched edges are included.

    Nodes and edges are prefixed with their source graph ('g1_' or 'g2_') and
    tagged with a 'source' attribute to indicate origin.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph
    graph2 : nx.Graph
        Second graph
    unmatched_edges_g1 : set or list
        Edge tuples from graph1 that couldn't be matched
    unmatched_edges_g2 : set or list
        Edge tuples from graph2 that couldn't be matched

    Returns
    -------
    nx.Graph
        Surplus graph containing all unmatched edges and their endpoint nodes.
        Node IDs are prefixed with 'g1_' or 'g2_' to avoid collisions.
        Each node and edge has a 'source' attribute ('g1' or 'g2').
    """
    surplus = nx.Graph()

    sources = [
        ("g1", graph1, unmatched_edges_g1),
        ("g2", graph2, unmatched_edges_g2),
    ]

    for prefix, graph, unmatched_edges in sources:
        for edge in unmatched_edges:
            u, v = edge[0], edge[1]
            # Add endpoint nodes if not already present
            for node in (u, v):
                node_id = f"{prefix}_{node}"
                if node_id not in surplus:
                    attr = dict(graph.nodes[node])
                    attr["source"] = prefix
                    attr["original_id"] = node
                    # Create v_coords from X, Y, Z if not present
                    attr["v_coords"] = np.array([float(attr["Z"]), float(attr["Y"]), float(attr["X"])])
                    surplus.add_node(node_id, **attr)

            # Add the edge
            edge_attr = dict(graph.edges[(u, v)])
            edge_attr["source"] = prefix
            edge_attr["original_edge"] = str((u, v))
            surplus.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **edge_attr)

    return surplus


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

all_stats, all_matches = compare_representations(g_gt, comparison_graphs, distance=distance)

# Visualize a specific comparison interactively (change label as needed)
viz_label = "i9"
g2_viz, fm_viz = all_matches[viz_label]
print_match_summary(g_gt, g2_viz, fm_viz)
viz_final_matches(g_gt, g2_viz, fm_viz)
print(1)