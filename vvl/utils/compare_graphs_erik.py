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


def get_edge_direction(G, edge):
    """Get the 3D direction vector of an edge."""
    u, v = edge
    pos_u = np.array([G.nodes[u]["X"], G.nodes[u]["Y"], G.nodes[u]["Z"]])
    pos_v = np.array([G.nodes[v]["X"], G.nodes[v]["Y"], G.nodes[v]["Z"]])
    direction = pos_v - pos_u
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


def angular_similarity(G1, edge1, G2, edge2):# -> Any:
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

    # Get coordinates along the edge
    edge_coords = get_edge_points(g1, edge)

    # Find nearby nodes in g2
    candidate_nodes = find_nearby_nodes(edge_coords, g2, max_distance)

    if len(candidate_nodes) < 2:
        return set()

    # Build subgraph with only candidate nodes
    subgraph = build_candidate_subgraph(g2, candidate_nodes)

    if len(subgraph.edges) == 0:
        return set()

    # Find longest path in subgraph
    longest_path = find_longest_path_in_subgraph(subgraph)

    if len(longest_path) < 2:
        return set()


    # TODO: Maybe dont just get longest path but best angular matching path as well...
    
    # Crop to match original edge endpoints
    start_coord = get_node_coords(g1, u)
    end_coord = get_node_coords(g1, v)
    cropped_path = crop_path_to_endpoints(longest_path, g2, start_coord, end_coord)

    if len(cropped_path) < 2:
        return set()

    if angular_similarity(g1, (u,v), g2, (cropped_path[0],cropped_path[1])) > 15:
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


g1 = load_graph(graph_gt)
g2 = load_graph(graph_i9)

distance = 15
matched_es1 = set()
matched_es2 = set()
matched_es1_dict = dict()
matched_es2_dict = dict()

# Find edges in g2 that match edges in g1
for e1 in tqdm(g1.edges, desc="Matching g1 edges to g2"):
    matched = get_matched_edges_for_edge(e1, g1, g2, max_distance=distance)
    if matched:
        matched_es1_dict[e1] = matched

# Find edges in g1 that match edges in g2
for e2 in tqdm(g2.edges, desc="Matching g2 edges to g1"):
    matched = get_matched_edges_for_edge(e2, g2, g1, max_distance=distance)
    if matched:
        matched_es2_dict[e2] = matched

to_delete_es1 = set()
for e1 in matched_es1_dict:
    for e2_matches in matched_es2_dict.values():
        if e1 in e2_matches or (e1[1], e1[0]) in e2_matches:
            # If e1 is connecting multiple vessels then only throw it out if the other match is longer or connecting more vessels
            if len(matched_es1_dict[e1]) > 1:
                if len(e2_matches) == len(matched_es1_dict[e1]): # If equal number of edges connected keep longer pair
                    e1_len = np.sum([get_edge_length(g2, e[0], e[1]) for e in matched_es1_dict[e1]])
                    e2_len = np.sum([get_edge_length(g1, e[0], e[1]) for e in e2_matches])
                    if e1_len > e2_len:
                        continue
                if len(e2_matches) <= len(matched_es1_dict[e1]): # Keep if longer
                    continue

            to_delete_es1.add(e1)
            break
for k in to_delete_es1:
    del matched_es1_dict[k]


to_delete_es2 = set()
for e1 in matched_es2_dict:
    for e2_matches in matched_es1_dict.values():
        if e1 in e2_matches or (e1[1], e1[0]) in e2_matches:
            # If e1 is connecting multiple vessels then only throw it out if the other match is longer
            if len(matched_es2_dict[e1]) > 1:
                if len(e2_matches) == len(matched_es2_dict[e1]):
                    e1_len = np.sum([get_edge_length(g1, e[0], e[1]) for e in matched_es2_dict[e1]])
                    e2_len = np.sum([get_edge_length(g2, e[0], e[1]) for e in e2_matches])
                    if e1_len > e2_len:
                        continue
                if len(e2_matches) <= len(matched_es2_dict[e1]): # Keep if longer
                    continue
            to_delete_es2.add(e1)
            break
for k in to_delete_es2:
    del matched_es2_dict[k]


get_matched_edges_for_edge((23,79), g1, g2, max_distance=distance) # = {(30, 65)}

matched_es1 = set(list(matched_es1_dict.keys()))
for ele in matched_es2_dict.values():
    matched_es1.update(ele)
matched_es2 = set(list(matched_es2_dict.keys()))
for ele in matched_es1_dict.values():
    matched_es2.update(ele)

# Compute unmatched edges (edges that weren't matched by the other graph)
unmatched_es1 = set()
unmatched_es2 = set()
for e1 in g1.edges:
    if e1 not in matched_es1 and (e1[1], e1[0]) not in matched_es1:
        unmatched_es1.add(e1)
for e2 in g2.edges:
    if e2 not in matched_es2 and (e2[1], e2[0]) not in matched_es2:
        unmatched_es2.add(e2)
surplus_graph = create_unmatched_edges_graph(g1, g2, unmatched_es1, unmatched_es2)
viz_matched_unmatched(g1, g2, matched_es1, matched_es2, unmatched_es1, unmatched_es2)
# ged = nx.graph_edit_distance(g1,g2)
print("Done")

new_es1  = set()
new_es2  = set()
for i,es1 in enumerate(matched_es1_dict.keys()):
    # new_es1.add(es1)
    # new_es2.update(matched_es1_dict[es1])
    viz_matched_unmatched(g1, g2, {es1}, matched_es1_dict[es1], set(), set())
    if i % 10 == 0:
        print(1)