# Graph Comparison Module

`compare_graphs.py` - A module for comparing two similar vascular graphs and identifying structural and feature differences.

## Purpose

When processing the same vascular volume with different parameters, algorithms, or manual corrections, you get slightly different graph representations. This module answers:

- Which vessels are present in both graphs?
- How do their features (radius, length, tortuosity) differ?
- Which vessels are missing from one graph?
- Where did one edge get split into multiple segments (or vice versa)?

## Design Approach

### Three-Stage Matching

The comparison uses a hierarchical matching strategy:

```
1. Node Pairing (spatial proximity)
        ↓
2. Edge Pairing (endpoint matching)
        ↓
3. Split/Merge Detection (path-based matching)
```

**Stage 1: Node Pairing**

Nodes are matched using the Hungarian algorithm on a distance cost matrix. This finds the globally optimal 1:1 assignment that minimizes total distance between paired nodes.

- Nodes beyond `max_node_distance` threshold remain unmatched
- Uses spatial coordinates (X, Y, Z) stored in GraphML

**Stage 2: Edge Pairing**

Once nodes are paired, edges are matched if both endpoints map to the corresponding endpoints in the other graph. This is strict but fast.

**Stage 3: Split/Merge Detection**

Edges that fail Stage 2 might still correspond to each other. Common cases:

- **Split**: One edge in graph1 corresponds to 2+ edges in graph2 (an intermediate branch point was detected)
- **Merge**: Multiple edges in graph1 correspond to one edge in graph2 (branch point was missed)

Detection works by:
1. For each unmatched edge, find the path between its matched endpoints in the other graph
2. Sample points along both the single edge and the multi-edge path
3. Compute spatial overlap using bidirectional KDTree coverage
4. If overlap exceeds threshold, record as split/merge match

### Output Graphs

**Difference Graph**: Contains all matched elements with comparison attributes:
- Original features from graph1
- `{feature}_g1`, `{feature}_g2` - values from each graph
- `{feature}_diff`, `{feature}_pct_diff` - absolute and percentage differences
- `paired_node_g2`, `paired_edge_g2` - IDs of matched elements

**Surplus Graph**: Contains unmatched elements from both graphs:
- Nodes/edges tagged with `source` attribute ("g1" or "g2")
- Preserves all original attributes
- Node IDs prefixed with source (e.g., "g1_42", "g2_15")

## Usage

### As a Library

```python
from vvl.compare_graphs import compare_graphs, load_graph, print_detailed_report

# Load graphs
g1 = load_graph("baseline.graphml")
g2 = load_graph("modified.graphml")

# Compare with default parameters
result = compare_graphs(g1, g2, max_node_distance=5.0)

# Access results
print(f"Matched edges: {len(result.edge_pairs)}")
print(f"Splits detected: {len([s for s in result.edge_splits if s.match_type == 'split'])}")

# Get edges with largest radius differences
from vvl.compare_graphs import get_largest_differences
top_diffs = get_largest_differences(result, feature="radius_avg", top_n=10)

# Print full report
print_detailed_report(result, g1, g2)

# Save output graphs
from vvl.compare_graphs import save_comparison_results
save_comparison_results(result, "diff.graphml", "surplus.graphml")
```

### From Command Line

```bash
python -m vvl.compare_graphs graph1.graphml graph2.graphml --detailed

# With custom parameters
python -m vvl.compare_graphs graph1.graphml graph2.graphml \
    --max-distance 3.0 \
    --output-diff differences.graphml \
    --output-surplus unmatched.graphml \
    --top-n 10
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_node_distance` | 5.0 | Maximum distance (in coordinate units) to pair nodes |
| `detect_splits` | True | Whether to run path-based split/merge detection |
| `overlap_threshold` | 0.7 | Minimum path overlap (0-1) to consider a split/merge match |
| `path_distance_threshold` | 3.0 | Distance threshold for overlap computation |
| `max_split_hops` | 4 | Maximum edges in a split path |

## Data Structures

### GraphComparisonResult

```python
@dataclass
class GraphComparisonResult:
    difference_graph: nx.Graph      # Paired elements with differences
    surplus_graph: nx.Graph         # Unmatched elements
    node_pairs: dict                # {node_g1: node_g2}
    edge_pairs: dict                # {(u1,v1): (u2,v2)}
    edge_splits: list[EdgeSplitMatch]  # Split/merge correspondences
    unmatched_nodes_g1: set
    unmatched_nodes_g2: set
    unmatched_edges_g1: set
    unmatched_edges_g2: set
```

### EdgeSplitMatch

```python
@dataclass
class EdgeSplitMatch:
    edges_g1: list      # Edge(s) from graph1
    edges_g2: list      # Edge(s) from graph2
    overlap_score: float  # Path overlap (0-1)
    match_type: str     # "split" or "merge"
```

## Limitations

1. **No registration**: Assumes graphs are in the same coordinate system. If there's translation, rotation, or scale difference, run alignment first.

2. **1:1 node matching**: The Hungarian algorithm assumes each node matches at most one other node. If a bifurcation was detected as one node in g1 but two nodes in g2, one will be unmatched.

3. **GraphML format loses path detail**: Saved GraphML files don't include `coords_list` (full edge path coordinates). Path overlap uses linear interpolation between endpoints as fallback.

4. **Hard distance thresholds**: Nodes at 5.01 units won't match with `max_node_distance=5.0` even if they're clearly the best candidate.

## Compared Features

Default features compared on edges:
- `length` - Physical length of vessel segment
- `radius_avg` - Average radius along segment
- `radius_max` - Maximum radius
- `radius_min` - Minimum radius
- `tortuosity` - Path length / straight-line distance
- `volume` - Cylindrical volume estimate
- `surface_area` - Lateral surface area

## Example Output

```
============================================================
Graph Comparison Summary
============================================================

Matched nodes: 847
Unmatched nodes (graph1): 12
Unmatched nodes (graph2): 23

Matched edges (1:1): 892
Edge splits (1->N): 8
Edge merges (N->1): 3
Unmatched edges (graph1): 15
Unmatched edges (graph2): 31

Difference graph: 847 nodes, 892 edges
Surplus graph: 67 nodes, 46 edges

------------------------------------------------------------
Feature Difference Statistics (paired edges)
------------------------------------------------------------

radius_avg:
  Mean diff: 0.0234
  Std diff:  0.1892
  Max diff:  1.4521
  Mean %diff: 3.42%
```
