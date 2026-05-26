# compare_graphs_erik.py

Bidirectional edge-matching between two VesselVio vessel graphs, plus visualization utilities.

---

## Overview

The module matches edges across two NetworkX graphs (e.g. ground-truth vs. reconstructed) using a spatial proximity + path-search strategy. It is also runnable as a standalone script that loads two `.pkl` graph files and prints final matches.

---

## I/O Helpers

| Function | Description |
|---|---|
| `load_graph(input_path)` | Load an igraph `.pkl` file and convert to `nx.Graph`. |
| `save_graph(g, output_path)` | Strip heavy attributes (`coords_list`, `radii_list`, etc.) and write as GraphML. |

---

## Geometry Utilities

| Function | Description |
|---|---|
| `get_node_coords(graph, node_id)` | Return `[X, Y, Z]` as `np.ndarray`. |
| `get_all_node_coords(graph)` | Return `(node_ids, coords_array)` for all nodes. |
| `get_edge_direction(G, edge)` | Normalized direction vector `(v - u) / ||v - u||`. |
| `angular_similarity(G1, edge1, G2, edge2)` | Angle in degrees between two edges (0 = parallel). Ignores direction. |
| `get_edge_points(G, edge)` | Return `coords_list` of intermediate points along an edge (parsed from string if needed). |
| `get_edge_length(g, u, v)` | Physical length: sum of segment lengths from `coords_list`, or Euclidean fallback. |
| `path_length(path, subgraph)` | Total physical length of a node-ID path. |

---

## Subgraph / Candidate Utilities

| Function | Description |
|---|---|
| `find_nearby_nodes(edge_coords, graph, max_distance)` | KDTree query: all nodes within `max_distance` of any point in `edge_coords`. |
| `build_candidate_subgraph(graph, candidate_nodes)` | Induced subgraph over `candidate_nodes` (only edges where both endpoints are candidates). |
| `find_longest_path_in_subgraph(subgraph)` | DFS from every degree-1 node; returns the physically longest path as a list of node IDs. |
| `find_length_matched_path_in_subgraph(subgraph, start, end, source_len)` | DFS from `start`; returns the path whose physical length is closest to `source_len`. If the end node is close (< 5 vx) a shortest-path attempt is made first. |
| `crop_path_to_endpoints(path, graph, start_coord, end_coord)` | Trim a path so its first and last nodes are closest to the given 3-D coordinates. |

---

## Core Matching

### `get_matched_edges_for_edge(edge, g1, g2, max_distance=5.0)`

Finds the set of edges in `g2` that spatially correspond to a single edge in `g1`.

**Algorithm:**
1. Extract `coords_list` of the source edge.
2. Find all nodes in `g2` within `max_distance` of any coordinate (via KDTree).
3. Build a candidate subgraph of `g2`.
4. Identify the candidate node nearest to the source edge's start and end.
5. Run `find_length_matched_path_in_subgraph` from the better-anchored endpoint.
6. Crop the path to align with the source endpoints.
7. Reject if the angular difference between source edge and matched path exceeds 30 degrees.
8. Return the set of `(u, v)` edge tuples forming the matched path.

Returns an empty set if no valid match is found.

---

### Bidirectional matching (script section)

```python
# Forward: for each edge in g1, find matching edges in g2
for e1 in g1.edges:
    matched_es1_dict[e1] = get_matched_edges_for_edge(e1, g1, g2, max_distance=distance)

# Reverse: for each edge in g2, find matching edges in g1
for e2 in g2.edges:
    matched_es2_dict[e2] = get_matched_edges_for_edge(e2, g2, g1, max_distance=distance)
```

### Final match consolidation

Iterates `matched_es1_dict` and cross-references `matched_es2_dict`:

- If forward and reverse matches agree (edge appears in both), the pair is kept as a confirmed match.
- If they disagree, the match with smaller combined endpoint distance (`get_node_dists_from_edge`) is preferred.
- Edges in `matched_es1_dict` with no reverse support are kept as unilateral matches.
- TODO: Currently there may be matches in matched_es2_dict that aren't included in final_matches. I need to identify these edges and determine which ones haven't been detected yet and add them

Result stored in `final_matches` (set of `(g1_edges, g2_edges)` tuples).

---

## Surplus Graph

### `create_unmatched_edges_graph(graph1, graph2, unmatched_edges_g1, unmatched_edges_g2)`

Combines unmatched edges from both graphs into a single `nx.Graph`. Node IDs are prefixed `g1_` / `g2_` to avoid collisions. Each node and edge carries a `source` attribute.

---

## Visualization

| Function | Description |
|---|---|
| `viz_graph(G)` | PyVista render of a merged graph; edges colored red (g1) or blue (g2) by `source` attribute. |
| `viz_matched_unmatched(...)` | Static 4-color PyVista render: G1 Matched (green), G2 Matched (blue), G1 Unmatched (red), G2 Unmatched (orange). |
| `viz_matched_unmatched_interactive(...)` | Same render but with left-click picking: prints edge info to console and opens a detail window showing the matched edge pair. |

### Color scheme

| Category | Color |
|---|---|
| G1 Matched | `#2E8B57` sea green |
| G2 Matched | `#4169E1` royal blue |
| G1 Unmatched | `#DC143C` crimson |
| G2 Unmatched | `#FF8C00` dark orange |

---

## Key Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_distance` | `5.0` (or `13` in script) | Max voxel distance for a node in g2 to be considered a candidate for a g1 edge. |
| `angle_threshold` (implicit) | `30 deg` | Max angular difference between source edge and matched path. |
| `end_dist_threshold` | `5 vx` | If the nearest end-node is within this distance, try `nx.shortest_path` first. |

---

## Script Usage

Edit the four hardcoded graph paths at the bottom of the file, then run:

```bash
python compare_graphs_erik.py
```

This loads `g1` (ground truth) and `g2` (reconstruction), runs bidirectional matching, and populates `final_matches`.
