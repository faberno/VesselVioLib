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


def save_graph(g,output_path):
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

def load_graph(filepath: str) -> nx.Graph:
    """Load a graph from GraphML file."""
    G = nx.read_graphml(filepath)
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)
    return G

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
    plotter.show()


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
            n: np.array([
                float(graph.nodes[n]["X"]),
                float(graph.nodes[n]["Y"]),
                float(graph.nodes[n]["Z"])
            ])
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
        "matched_g1": "#2E8B57",    # Sea green
        "matched_g2": "#4169E1",    # Royal blue
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
    plotter.show()    

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


def angular_similarity(G1, edge1, G2, edge2):
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


def check_proximity(G1, edge1, G2, edge2, max_distance=5.0):
    """
    Check if all points on the shorter edge are within max_distance
    of at least one point on the longer edge.

    Returns:
        is_similar: bool - True if all points satisfy the criterion
        max_min_dist: float - the maximum of the minimum distances
        distances: array - minimum distance for each point on shorter edge
    """
    points1 = get_edge_points(G1, edge1)
    points2 = get_edge_points(G2, edge2)

    # Determine shorter and longer
    len1 = np.linalg.norm(points1[-1] - points1[0])
    len2 = np.linalg.norm(points2[-1] - points2[0])

    if len1 <= len2:
        shorter, longer = points1, points2
    else:
        shorter, longer = points2, points1

    # For each point on shorter, find min distance to any point on longer
    min_distances = np.zeros(len(shorter))
    for i, pt in enumerate(shorter):
        dists = np.linalg.norm(longer - pt, axis=1)
        min_distances[i] = np.min(dists)

    max_min_dist = np.max(min_distances)
    is_similar = max_min_dist <= max_distance

    return is_similar, max_min_dist, min_distances, len1, len2


def compare_edges(G1, edge1, G2, edge2, angle_threshold=15.0, distance_threshold=5.0, edge_case_small_len=10.0):
    """
    Full comparison of two edges.

    Returns dict with:
        - angle_deg: angle between edges in degrees
        - angles_similar: True if angle <= threshold
        - proximity_satisfied: True if all points within distance
        - max_min_distance: worst-case minimum distance
        - is_similar: True if both criteria met
    """
    angle = angular_similarity(G1, edge1, G2, edge2)
    prox_ok, max_min_dist, min_dists, len_e1, len_e2 = check_proximity(G1, edge1, G2, edge2, distance_threshold)

    is_similar = 1.0 if (angle <= angle_threshold) and prox_ok else 0.0
    # If proximity is good then edge_similarity may be okay as long as one of the vessels is extremely small.
    if is_similar == 0.0 and prox_ok and (len_e1 < edge_case_small_len or len_e2 < edge_case_small_len):
        is_similar = 0.5

    return {
        "angle_deg": angle,
        "angles_similar": angle <= angle_threshold,
        "proximity_satisfied": prox_ok,
        "max_min_distance": max_min_dist,
        "min_dists": min_dists,
        "edge_lengths": (len_e1, len_e2),
        "is_similar": is_similar,
    }


def get_similar_edges(e1, g1, g2):
    similars = dict()
    for e2 in g2.edges:
        similarity = compare_edges(g1, e1, g2, e2)
        if similarity["is_similar"] > 0:
            similars[e2] = similarity
    return similars 


def determine_connections(similarities):# -> dict[Any, Any]:
    """
    For each connected component, find the longest path and return
    only the similarities for edges on those longest paths.
    """
    es = list(similarities.keys())
    ns = set([e[0] for e in es] + [e[1] for e in es])
    
    g = nx.Graph()
    g.add_nodes_from(ns)
    g.add_edges_from(es)
    
    components = list(nx.connected_components(g))
    
    # Collect edges from longest path in each component
    longest_path_edges = set()
    
    for component in components:
        subgraph = g.subgraph(component).copy()
        longest_path = find_longest_path(subgraph, similarities)
        
        # Convert path (node list) to edges
        for i in range(len(longest_path) - 1):
            u, v = longest_path[i], longest_path[i + 1]
            # Store in canonical order to match similarities keys
            edge = (u, v) if (u, v) in similarities else (v, u)
            longest_path_edges.add(edge)
    
    # Filter similarities to only longest path edges
    return {e: v for e, v in similarities.items() if e in longest_path_edges}


def find_longest_path(g, similarities):
    """
    Find the longest path in a graph, using edge weights from similarities.
    Works for graphs with bifurcations by exploring all endpoint-to-endpoint paths.
    """
    # Find endpoints (degree 1) and bifurcations (degree > 2)
    endpoints = [n for n in g.nodes if g.degree(n) == 1]
    
    # If no endpoints (cycle), pick arbitrary start
    if not endpoints:
        endpoints = [list(g.nodes)[0]]
    
    def get_edge_weight(u, v):
        """Get path length from similarities, default to 1."""
        for key in [(u, v), (v, u)]:
            if key in similarities:
                sim = similarities[key]
                # Use a length metric - adapt based on your similarities structure
                if isinstance(sim, dict) and 'length' in sim:
                    return sim['length']
                elif isinstance(sim, dict) and 'max_min_distance' in sim:
                    return 1  # or use some other metric
        return 1
    
    def path_length(path):
        """Total length of a path."""
        total = 0
        for i in range(len(path) - 1):
            total += get_edge_weight(path[i], path[i + 1])
        return total
    
    # Find longest path using DFS from each endpoint
    longest = []
    longest_len = 0
    
    for start in endpoints:
        # DFS to find all paths to other endpoints
        stack = [(start, [start], set([start]))]
        
        while stack:
            node, path, visited = stack.pop()
            
            # Check if this is an endpoint (other than start) or dead end
            neighbors = [n for n in g.neighbors(node) if n not in visited]
            
            if not neighbors:
                # End of path - check if longest
                plen = path_length(path)
                if plen > longest_len:
                    longest = path
                    longest_len = plen
            else:
                for neighbor in neighbors:
                    stack.append((neighbor, path + [neighbor], visited | {neighbor}))
    
    return longest


def find_longest_path_simple(g):
    """
    Simpler alternative: longest path = diameter for tree-like structures.
    Uses BFS twice to find the longest shortest path.
    Only works correctly for trees (no cycles).
    """
    if len(g.nodes) == 0:
        return []
    
    # BFS from arbitrary node to find farthest node
    start = list(g.nodes)[0]
    distances = nx.single_source_shortest_path_length(g, start)
    farthest = max(distances, key=distances.get)
    
    # BFS from farthest to find the actual farthest pair
    distances = nx.single_source_shortest_path_length(g, farthest)
    other_end = max(distances, key=distances.get)
    
    # Get the actual path
    return nx.shortest_path(g, farthest, other_end)
    

def detect_split_edges(e1,similarites,g):
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
        ("g2", graph2, unmatched_edges_g1),
        ("g1", graph1, unmatched_edges_g2),
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
                    attr["v_coords"] = np.array([
                        float(attr["Z"]),
                        float(attr["Y"]),
                        float(attr["X"])
                    ])
                    surplus.add_node(node_id, **attr)

            # Add the edge
            edge_attr = dict(graph.edges[(u, v)])
            edge_attr["source"] = prefix
            edge_attr["original_edge"] = str((u, v))
            surplus.add_edge(f"{prefix}_{u}", f"{prefix}_{v}", **edge_attr)

    return surplus


graph_gt = r"E:\sr_data\532\cuff_analysis\graph_comparison_cuff_base\original\vesselvio\Graphs\R_20190216163532_AngelosHyperamia_base_1_RSOM50_wl1_corr_v_rgb_pred.pkl"
graph_i25 = r"E:\sr_data\532\cuff_analysis\graph_comparison_cuff_base\i25\vesselvio\Graphs\R_20190216163532_AngelosHyperamia_base_1_RSOM50_wl1_corr_v_rgb_pred.pkl"

with open(graph_gt, "rb") as f:
    g1 = pickle.load(f)
    g1 = nx.Graph(g1.to_networkx())
with open(graph_i25, "rb") as f:
    g2 = pickle.load(f)
    g2 = nx.Graph(g2.to_networkx())

matched_es1 = set()
unmatched_es1 = set()
for e1 in tqdm(g1.edges):
    similars = get_similar_edges(e1, g1, g2)
    if len(similars) != 0:
        similars_connected = determine_connections(similars)
        for e in similars_connected.keys():
            matched_es1.add(e)

matched_es2 = set()
unmatched_es2 = set()
for e1 in tqdm(g2.edges):
    similars = get_similar_edges(e1, g2, g1)
    if len(similars) != 0:
        similars_connected = determine_connections(similars)
        for e in similars_connected.keys():
            matched_es2.add(e)

for e1 in g2.edges:
    if e1 not in matched_es1:
        unmatched_es1.add(e1)
for e1 in g1.edges:
    if e1 not in matched_es2:
        unmatched_es2.add(e1)


surplus_graph = create_unmatched_edges_graph(g1, g2, matched_es1, matched_es2)
output_path = os.path.join(os.path.dirname(graph_gt), "Graphs", "test1.graphml")
viz_matched_unmatched(g1,g2,matched_es1,matched_es2,unmatched_es1,unmatched_es2)
print("wow")
