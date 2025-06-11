"""
Graph construction and processing pipelines
"""

__author__ = "Jacob Bumgarner <jrbumgarner@mix.wvu.edu>"
__license__ = "GPLv3 - GNU General Pulic License v3 (see LICENSE)"
__copyright__ = "Copyright 2022 by Jacob Bumgarner"
__webpage__ = "https://jacobbumgarner.github.io/VesselVio/"
__download__ = "https://jacobbumgarner.github.io/VesselVio/Downloads"


from multiprocessing import cpu_count
from time import perf_counter as pf
import logging

from vvl.utils import measure_time, GraphType, is_unix

logger = logging.getLogger(__name__)

import igraph as ig
import numpy as np


from numba import njit


#######################
### Graph Reduction ###
#######################
def simplify_graph(
    g,
    reduced_edges,
    volumes,
    surface_areas,
    lengths,
    tortuosities,
    radii_avg,
    radii_max,
    radii_min,
    radii_SD,
    vis_radii,
    coords_lists,
    radii_lists,
):
    g.delete_edges(g.es())
    g.add_edges(
        reduced_edges,
        {
            "volume": volumes,
            "surface_area": surface_areas,
            "length": lengths,
            "tortuosity": tortuosities,
            "radius_avg": radii_avg,
            "radius_max": radii_max,
            "radius_min": radii_min,
            "radius_SD": radii_SD,
            "coords_list": coords_lists,
            "radii_list": radii_lists,
            "vis_radius": vis_radii,
        },
    )
    g.delete_vertices(g.vs.select(_degree=0))
    return


######################
### Edge Detection ###
######################

## Scanning orientations for edge detection.
def orientations():
    # Prepration for edge point analysis. Directional edge detection
    # based on C. Kirst's algorithm.
    # Matrix preparation for edge detetction
    # Point of interest rests at [1,1,1] in 3x3x3 array.
    scan = np.array(
        [
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],  # End of top slice
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [1, 1, 2],  # End of middle slice
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
        ]
    )  # End of bottom slice

    scan -= 1

    return scan


## Construct vertex index LUT
def construct_vLUT(points: np.ndarray, volume_shape: tuple):
    """DOCTODO

    Args:
        points:
        volume_shape:

    Returns:

    """
    values = np.arange(0, points.shape[0])
    vertex_LUT = np.zeros(volume_shape, dtype=np.int_)
    vertex_LUT[points[:, 0], points[:, 1], points[:, 2]] = values
    return vertex_LUT


@njit(cache=True)
def identify_edges(points: np.ndarray, vertex_LUT: np.ndarray, spaces: np.ndarray) -> list:
    """Return of all neighboring centerline points."""
    edges = []

    for i in range(points.shape[0]):
        local = spaces + points[i]

        for j in range(local.shape[0]):
            # Check if neighbor is a non-zero, then add edge if it is.
            target_index = vertex_LUT[local[j, 0], local[j, 1], local[j, 2]]
            if target_index > 0:
                edges.append((i, target_index))
    return edges


#########################
### Clique Processing ###
#########################

@measure_time
def remove_cliques(g):
    """Filter the graph for cliques."""

    delete_list = []

    while True:

        cliques = g.maximal_cliques(min=3)

        if len(cliques) == 0:
            break

        for clique in cliques:
            g_clique = g.vs[clique]

            # Weight the vs based on radius and neighbor radius
            weights = g_clique["v_radius"]
            for i, v in enumerate(g_clique):
                for n in v.neighbors():
                    weights[i] += n["v_radius"]

            # Sort the vertices based on their weights, remove edge between smallest
            sorted_ids = [v.index for _, v in sorted(zip(weights, g_clique))]
            edge = (sorted_ids[0], sorted_ids[1])
            delete_list.append(edge)

        g.delete_edges(delete_list)



#######################
### Segment Pruning ###
#######################

def segment_isolation(g, degree_filter, prune_length):
    """ Isolate segments from the graph based on a degree filter."""
    # Select all vertices with a degree less than the filter value.
    # Those vertices represent the segment part of the vessels and exclude branch points.
    segment_ids = g.vs.select(_degree_lt=degree_filter)
    g_segments = g.subgraph(segment_ids)
    segments = [s for s in g_segments.clusters() if len(s) < max(1, prune_length)]  # todo: doesnt this filter all segments and not just endpoints??
    return g_segments, segments, segment_ids


# Remove short connected endpoint segments from the main graph.
# g, resolution, prune_length, verbose=False
def segment_pruning(bottom, top):
    # Prune only the connected endpoints here.
    # Isolated segments are pruned later with flood filling in VolProc.
    vertices_togo = []
    pruned = 0
    for segment in segments[bottom:top]:
        num_verts = len(segment)
        if num_verts < g_prune_len:
            # Isolate endpoint segments. Should only have one vertex with degree == 1
            # Faster than calling .indices
            vertices = [segment_ids[vertex].index for vertex in segment]
            degrees = g.degree(vertices)
            ends = degrees.count(1)

            # If endpoint segment, calculate the size.
            if ends == 1:
                # Send off to our feature extraction to find the size
                if num_verts == 1:
                    segment_length = FeatExt.small_seg_path(
                        g,
                        segment,
                        segment_ids,
                        g_res,
                        centerline_smoothing=g_cl_smoothing,
                        pruning=True,
                    )

                elif num_verts > 1:
                    segment_length = FeatExt.large_seg_path(
                        g,
                        gsegs,
                        segment,
                        segment_ids,
                        g_res,
                        centerline_smoothing=g_cl_smoothing,
                        pruning=True,
                    )

                if segment_length < g_prune_len:
                    pruned += 1
                    vertices_togo.extend(vertices)

    return [vertices_togo, pruned]


def v_graph_pruning_io():
    pruned = 0
    vertices_togo = []

    workers = cpu_count()
    seg_count = len(segments)
    if helpers.unix_check() and seg_count > workers:
        results = helpers.multiprocessing_input(
            segment_pruning, seg_count, workers, sublist=True
        )
        for result in results:
            vertices_togo.extend(result[0])
            pruned += result[1]
    else:
        vertices_togo, pruned = segment_pruning(0, seg_count)

    g.delete_vertices(vertices_togo)
    return pruned


# Prune endpoints of edge-graphs
def edge_graph_prune(g, segment_ids, segments, prune_length):
    pruned = 0
    vertices_togo = []
    for segment in segments:
        if len(segment) > 1:
            continue  # Just in case something goes wrong with
        edge = g.incident(segment_ids[segment[0]].index)
        if len(edge) > 1:
            continue  # Fail safe again

        segment_length = g.es[edge[0]]["length"]
        if segment_length < prune_length:
            vertices_togo.append(segment_ids[segment[0]].index)
            pruned += 1

    g.delete_vertices(vertices_togo)
    return pruned


def filter_graph_edges(
        g: ig.Graph,
        minimum_endpoint_segment_length: float,
        minimum_isolated_segment_length: float,
        resolution: np.ndarray,
        centerline_smoothing: bool = True,
        graph_type: GraphType = GraphType.CENTERLINE,
):
    if minimum_endpoint_segment_length > 0.0:
        prune_short_graph_endpoints(
            g,
            minimum_endpoint_segment_length,
            resolution,
            centerline_smoothing=centerline_smoothing,
            graph_type=graph_type
        )

    if minimum_isolated_segment_length > 0.0:
        filter_input(
            g,
            minimum_isolated_segment_length,
            resolution,
            centerline_smoothing=centerline_smoothing,
            graph_type=graph_type,
        )




def prune_short_graph_endpoints(
    g: ig.Graph,
    prune_length: float,
    resolution: np.ndarray,
    centerline_smoothing: bool =True,
    graph_type = GraphType.CENTERLINE
):

    if graph_type == GraphType.CENTERLINE:
        degree_filter = 3
    else:
        degree_filter = 2

    gsegs, segments, segment_ids = segment_isolation(g, degree_filter, prune_length)

    if graph_type == GraphType.CENTERLINE:
        v_graph_pruning_io()

        g_prune_len = 1.01
        gsegs, segments, segment_ids = segment_isolation(g, degree_filter)
        p2 = v_graph_pruning_io()

    else:
        p1 = edge_graph_prune(g, segment_ids, segments, prune_length)
        # No second pass for edge graphs
        p2 = 0

    return


#########################
### Segment Filtering ###
#########################
# Filter segments based on some length
def vgraph_segment_filter(bottom, top):
    vertices_togo = []
    filtered = 0

    # Iterate through clusters, identify segments,
    # filter those short enough to be removed
    for cluster in clusters[bottom:top]:
        # Check to see that we have an isolated segment, i.e., no branch points
        degrees = g.degree(cluster)
        cluster_length = len(cluster)
        if degrees.count(1) == 2:  # Only examine isolated segments
            if cluster_length < 4:
                segment_length = FeatExt.small_seg_path(
                    g,
                    cluster,
                    resolution=g_res,
                    centerline_smoothing=cl_smoothing,
                    pruning=True,
                )
            else:
                segment_length = FeatExt.large_seg_filter(
                    g, cluster, g_res, centerline_smoothing=cl_smoothing
                )

            if segment_length < g_filter_len:  # Remove the vertices if short enough
                vertices_togo.extend(cluster)
                filtered += 1

    return [vertices_togo, filtered]


def vgraph_segment_filter_io(g, filter_length, resolution, centerline_smoothing):
    # Set up globals for forked multiprocessing
    global clusters, g_filter_len, g_res, ret_coords, cl_smoothing
    g_filter_len = filter_length
    g_res = resolution.copy()
    cl_smoothing = centerline_smoothing

    filtered = 0
    vertices_togo = []

    # Label clusters in the dataset
    clusters = g.components()

    # If we are here, that means that the filter value is non-zero.
    # So find all clusters that are either 2 vertices long
    # or those that are shorter than the filter length, whichever is the largest
    clusters = [c for c in clusters if len(c) <= max(2, g_filter_len)]

    seg_count = len(clusters)
    workers = cpu_count()
    if helpers.unix_check() and seg_count > workers:
        results = helpers.multiprocessing_input(
            vgraph_segment_filter, seg_count, workers, sublist=True
        )
        for result in results:
            vertices_togo.extend(result[0])
            filtered += result[1]
    else:
        vertices_togo, filtered = vgraph_segment_filter(0, seg_count)

    g.delete_vertices(vertices_togo)

    # Global variable cleanup
    del (clusters, g_filter_len, g_res, cl_smoothing)

    return filtered


# Filter isolated segments in edge graphs
def egraph_segment_filter(g, filter_length):
    vertices_togo = []
    filtered = 0

    # Isolate individual segment clusters
    clusters = g.clusters()
    for cluster in clusters:
        if len(cluster) == 2:  # If segment is isoalted
            edge = g.incident(cluster[0])

            # Check edge length, delete if short enough
            if g.es[edge[0]]["length"] < filter_length:
                vertices_togo.extend(cluster)
                filtered += 1

    g.delete_vertices(vertices_togo)
    return filtered


def filter_input(
    g,
    filter_length,
    resolution,
    centerline_smoothing=True,
    graph_type="Centerlines",
    verbose=False,
):
    if verbose:
        t = pf()
        print("Filtering isolated segments...", end="\r")

    # Eliminate isolated vertices
    g.delete_vertices(g.vs.select(_degree=0))

    # Vertex graph segment filtering
    if filter_length > 0:
        if graph_type == "Centerlines":
            filtered = vgraph_segment_filter_io(
                g, filter_length, resolution, centerline_smoothing
            )

        # Edge graph segment filtering
        else:
            filtered = egraph_segment_filter(g, filter_length)

    if verbose:
        if filter_length > 0:
            print(
                f"Filtered {filtered} isolated " f"segments in {pf() - t:0.2f} seconds."
            )
        else:
            print("", end="\r")

    return


######################
### Graph creation ###
######################


class VesselGraph:
    def __init__(self, g: ig.Graph):
        self.centerline_graph = g.copy()
        self.branch_graph = self.simplify_centerline_graph()
        self.compute_edge_lengths()

    def simplify_centerline_graph(self):
        g = self.centerline_graph

        deg = g.degree()
        keep = {v.index for v in g.vs if deg[v.index] != 2}

        simplified_edges = []
        visited = set()

        for v in keep:
            for nbr in g.neighbors(v):
                if (v, nbr) in visited or (nbr, v) in visited:
                    continue
                path = [v]
                current = nbr
                prev = v
                while g.degree(current) == 2 and current not in keep:
                    path.append(current)
                    next_nodes = [n for n in g.neighbors(current) if n != prev]
                    if not next_nodes:
                        break  # dead end
                    prev, current = current, next_nodes[0]
                path.append(current)
                visited.add((v, current))
                visited.add((current, v))
                simplified_edges.append((path[0], path[-1]))

        # Build new graph
        new_vertices = sorted(keep)
        id_map = {old: new for new, old in enumerate(new_vertices)}
        new_g = ig.Graph()
        new_g.add_vertices(len(new_vertices))
        new_g.add_edges([(id_map[u], id_map[v]) for u, v in simplified_edges])

        # Add original_id attribute
        new_g.vs["original_id"] = new_vertices

        return new_g

    def compute_edge_lengths(self):
        print()


@measure_time
def create_graph(
    centerlines, centerline_radii, volume_shape
):

    # Create graph, populate graph with correct number of vertices.
    g = ig.Graph()
    g.add_vertices(len(centerlines))

    # Populate vertices with cartesian coordinates and radii
    g.vs["v_coords"] = centerlines
    g.vs["v_radius"] = centerline_radii

    # Prepare what we need for our edge identifictation
    spaces = orientations()  # 13-neighbor search
    vertex_LUT = construct_vLUT(centerlines, volume_shape)
    edges = identify_edges(centerlines, vertex_LUT, spaces)
    g.add_edges(edges)

    # Remove spurious branchpoints from our labeling
    remove_cliques(g)

    vessel_graph = VesselGraph(g)

    return vessel_graph
