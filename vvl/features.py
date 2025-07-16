from typing import Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from scipy.optimize import minimize_scalar, curve_fit
from scipy.signal import savgol_filter
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage
import skan
from skimage.morphology import skeletonize


def fractal_dimension(
        array: np.ndarray,
        max_box_size: int = None,
        min_box_size: int = 1,
        n_samples: int = 20,
        n_offsets: int = 0,
):
    """Calculates the fractal dimension of a 3D numpy array.
    Source: https://github.com/ChatzigeorgiouGroup/FractalDimension/tree/master
    Author: @DanielDondorp
    License: GPL-3.0 (no changes have been made to this function, except type hinting and the docstring format.)

    Parameters
    ----------
    array: np.ndarray
        The array to calculate the fractal dimension of.
    max_box_size: int
        The largest box size, given as the power of 2 so that
        2**max_box_size gives the sidelength of the largest box.
    min_box_size: int
        The smallest box size, given as the power of 2 so that
        2**min_box_size gives the sidelength of the smallest box.
        Default value 1.
    n_samples: int
        Number of scales to measure over.
    n_offsets: int N
        umber of offsets to search over to find the smallest set N(s) to
        cover  all voxels>0.
    plot: bool
        Set to true to see the analytical plot of a calculation.

    Returns
    -------
    float
        Fractal dimension of a 3D numpy array
    """
    # determine the scales to measure on
    if max_box_size == None:
        # default max size is the largest power of 2 that fits in the smallest dimension of the array:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
    scales = np.unique(scales)  # remove duplicates that could occur as a result of the floor

    # get the locations of all non-zero pixels
    locs = np.where(array > 0)
    voxels = np.array([(x, y, z) for x, y, z in zip(*locs)])
    if len(voxels) == 0:
        return 0

    # count the minimum amount of boxes touched
    Ns = []
    # loop over all scales
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        # search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(0, i, scale) for i in array.shape]
            bin_edges = [np.hstack([0 - offset, x + offset]) for x in bin_edges]
            H1, e = np.histogramdd(voxels, bins=bin_edges)
            touched.append(np.sum(H1 > 0))
        Ns.append(touched)
    Ns = np.array(Ns)

    # From all sets N found, keep the smallest one at each scale
    Ns = Ns.min(axis=1)

    # Only keep scales at which Ns changed
    scales = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])

    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[: len(Ns)]
    # perform fit
    coeffs = np.polyfit(np.log(1 / scales), np.log(Ns), 1)

    return {"fractal_dimension": coeffs[0]}

def median_bifurcation_exponent(G: nx.Graph, large_vessel_radius: float):
    """Calculates the median bifurcation exponent of the vessels.
    For every bifurcation, define the thickest vessel as parent node the remaining as children.
    Use Murrays Law to calculate the bifurcation exponent.

    Parameters
    ----------
    G: nx.Graph
        Networkx Graph of the vessel volume

    Returns
    -------
    float
        Median bifurcation exponent of the vessel volume

    """
    all_nodes = [node for node, degree in G.degree() if degree >= 3]
    node_edges = [G.edges(n, data=True) for n in all_nodes]

    all_bif_exp = []
    all_bif_exp_small = []
    all_bif_exp_large = []
    for edges in node_edges:
        if len(edges) < 3:
            continue
        radii = [e[2]["radius_avg"] for e in edges]
        radii.sort()
        parent = radii.pop()
        children = np.asarray(radii)

        def murray(x):
            a = np.sum(children ** x)
            b = a ** (1 / x)
            c = parent - b
            d = np.abs(c)
            return d
            # return np.abs(parent - np.sum(children ** x) ** (1 / x))

        try:
            bif = minimize_scalar(murray, bounds=(1.0, 100.0), method="Bounded")
            all_bif_exp.append(bif.x)
            if parent < large_vessel_radius:
                all_bif_exp_small.append(bif.x)
            else:
                all_bif_exp_large.append(bif.x)
        except:
            continue
    return {"median_bifurcation_exponent": np.nanmedian(all_bif_exp).item(),
            "median_large_vessel_bifurcation_exponent": np.nanmedian(all_bif_exp_large).item(),
            "median_small_vessel_bifurcation_exponent": np.nanmedian(all_bif_exp_small).item()}

def median_branch_length_ratio(G: nx.Graph, large_vessel_radius: float):
    all_nodes = [node for node, degree in G.degree() if degree >= 3]
    node_edges = [G.edges(n, data=True) for n in all_nodes]

    all_ratios = []
    all_ratios_small = []
    all_ratios_large = []
    for edges in node_edges:
        if len(edges) < 3:
            continue
        info = [(e[2]['length'], e[2]["radius_avg"]) for e in edges]
        info.sort(key=lambda x: x[1])  # Sort by radius
        parent = info.pop()

        all_ratios.append(parent[0] / parent[1])
        if parent[1] < large_vessel_radius:
            all_ratios_small.append(parent[0] / parent[1])
        else:
            all_ratios_large.append(parent[0] / parent[1])

    return {"median_branch_length_ratio": np.nanmedian(all_ratios).item(),
            "median_large_vessel_branch_length_ratio": np.nanmedian(all_ratios_large).item(),
            "median_small_vessel_branch_length_ratio": np.nanmedian(all_ratios_small).item()}

def blood_volume_features(G: nx.Graph, total_volume: float, large_vessel_radius: float):
    vessel_volumes = [(data["volume"], data["radius_avg"]) for _, _, data in G.edges(data=True)]

    total_blood_volume = sum(v[0] for v in vessel_volumes) / total_volume
    large_vessel_blood_volume = sum(v[0] for v in vessel_volumes if v[1] >= large_vessel_radius) / total_volume
    small_vessel_blood_volume = sum(v[0] for v in vessel_volumes if v[1] < large_vessel_radius) / total_volume

    return {"total_blood_volume": total_blood_volume,
            "small_vessel_blood_volume": small_vessel_blood_volume,
            "large_vessel_blood_volume": large_vessel_blood_volume}

def vessel_length_features(G: nx.Graph, large_vessel_radius: float, total_volume: float):
    """Calculates the total length of the vessel network (for all vessels, large vessels, small vessels), the median length of a vessel and the
    length of the longest vessel."""

    vessels = [data["length"] for _, _, data in G.edges(data=True)]
    small_vessels = [
        data["length"]
        for _, _, data in G.edges(data=True)
        if data["radius_avg"] < (large_vessel_radius)
    ]
    large_vessels = [
        data["length"]
        for _, _, data in G.edges(data=True)
        if data["radius_avg"] >= (large_vessel_radius)
    ]

    total_vessel_length = sum(vessels)
    total_small_vessel_length = sum(small_vessels)  # max RadiusAvg is probably the most accurate
    total_large_vessel_length = sum(large_vessels)

    median_vessel_length = 0.0 if len(vessels) == 0 else np.median(vessels).item()
    median_large_vessel_length = 0.0 if len(large_vessels) == 0 else np.median(large_vessels).item()
    median_small_vessel_length = 0.0 if len(small_vessels) == 0 else np.median(small_vessels).item()

    return {
        "total_vessel_length": total_vessel_length / total_volume,
        "total_large_vessel_length": total_large_vessel_length / total_volume,
        "total_small_vessel_length": total_small_vessel_length / total_volume,
        "median_vessel_length": median_vessel_length,  # todo: should this be normalized?
        "median_large_vessel_length": median_large_vessel_length,
        "median_small_vessel_length": median_small_vessel_length,
    }

def bifurcation_features(G: nx.Graph, total_volume: float, large_vessel_radius: float):
    """Calculates the number of bifurcations (in total and normalized by the image volume."""
    n_bifurcations = sum(1 for node, degree in G.degree() if degree > 2)

    out = {"#bifurcations": n_bifurcations / total_volume}
    out.update(median_bifurcation_exponent(G, large_vessel_radius))
    out.update(median_branch_length_ratio(G, large_vessel_radius))

    return out

def skan_features(image_volume: np.ndarray, total_volume: float):
    if image_volume.sum() == 0:
        return {
            "branch_number": 0.0,
            "branch_j2e_total": 0.0,
            "branch_j2j_total": 0.0,
            "num_junctions": 0.0,
        }

    info = skan.summarize(skan.Skeleton(skeletonize(image_volume, method='lee')),separator='-')
    branch_data = info.loc[info['branch-distance'] > 9]

    branch_number = len(branch_data['branch-distance'].values)
    branch_j2e_total = np.sum(branch_data['branch-type'].values == 1)
    branch_j2j_total = np.sum(branch_data['branch-type'].values == 2)
    num_junctions = np.unique(branch_data['node-id-src'].values).shape[0]

    return {
        "branch_number": branch_number / total_volume,
        "branch_j2e_total": branch_j2e_total / total_volume,
        "branch_j2j_total": branch_j2j_total / total_volume,
        "num_junctions": num_junctions / total_volume,
    }

def cycle_features(G: nx.Graph, total_volume: float):
    """Calculates the number of cycles, median cycle length and maximum cycle length of a graph."""
    # Convert multigraph to simple graph by keeping only one edge between nodes
    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    basis = nx.cycle_basis(G)
    return {
        "#cycles": len(basis) / total_volume,
        "median_cycle_length": 0.0 if len(basis) == 0 else np.median([len(b) for b in basis]), # todo: do these features need to be normalized?
        "max_cycle_length": 0.0 if len(basis) == 0 else max([len(b) for b in basis]),
    }

def component_length_features(G: nx.Graph, total_volume: float):
    """Calculates the median length per connected component, median distance per
    connected component and the length of the longest connected component."""
    lengths = []
    # distances = []
    for c in nx.connected_components(G):
        g = G.subgraph(c)
        # distances.append(sum([(data['distance']) for _, _, data in g.edges(data=True)]))
        lengths.append(sum([(data["length"]) for _, _, data in g.edges(data=True)]))
    return {
        "#components_experimental": len(lengths) / total_volume,
        "median_length_per_component_experimental": np.median(lengths), # todo: do these features need to be normalized?
        "longest_connected_component_experimental": np.max(lengths) / total_volume,
    }

def radius_features(G: nx.Graph, large_vessel_radius: float):
    radii = np.array([data["radius_avg"] for _, _, data in G.edges(data=True)])
    return {
        "median_radius": np.median(radii),
        "median_small_vessel_radius": np.median(radii[radii < large_vessel_radius]),
        "median_large_vessel_radius": np.median(radii[radii >= large_vessel_radius]),
    }

def extract_radius(G: nx.Graph):
    radii = np.array([data["radius_avg"] for _, _, data in G.edges(data=True)])
    return np.median(radii)

def graph_metric_features(G: nx.Graph, large_vessel_radius: float):
    density = nx.density(G)
    
    degree_assortativity_coefficient = (
        0
        if nx.degree_assortativity_coefficient(G) is None
        else nx.degree_assortativity_coefficient(G)
    )
    
    if G.number_of_nodes() == 0:
        mean_degree = 0.0
        mean_large_vessel_degree = 0.0
        mean_small_vessel_degree = 0.0
    else:
        radii_list = [np.array([e[2]['radius_avg'] for e in G.edges(node, data=True)]) for node in G.nodes()]

        degrees = []
        degrees_large_vessels = []
        degrees_small_vessels = []
        for radii in radii_list:
            degrees.append(len(radii))
            if np.any(radii >= large_vessel_radius):
                degrees_large_vessels.append(len(radii))
            if np.any(radii < large_vessel_radius):
                degrees_small_vessels.append(len(radii))

        mean_degree = np.mean(degrees)
        mean_large_vessel_degree = np.mean(degrees_large_vessels)
        mean_small_vessel_degree = np.mean(degrees_small_vessels)
    
    return {
        "density": density,
        "degree_assortativity_coefficient": degree_assortativity_coefficient,
        "mean_degree": mean_degree,
        "mean_large_vessel_degree": mean_large_vessel_degree,
        "mean_small_vessel_degree": mean_small_vessel_degree,
    }

def vessel_tortuosity_features(G: nx.Graph, large_vessel_radius: float, resolution: np.ndarray):
    """Calculates the median tortuosity of the vessels."""
    edge_info = [(data['radius_avg'], data["length"] / np.linalg.norm((data["coords_list"][0] - data["coords_list"][-1]) * resolution)) for _, _, data in G.edges(data=True)]
    mean_tortuosity = np.mean([e[1] for e in edge_info])
    mean_large_vessel_tortuosity = np.mean([e[1] for e in edge_info if e[0] >= large_vessel_radius])
    mean_small_vessel_tortuosity = np.mean([e[1] for e in edge_info if e[0] < large_vessel_radius])

    return {"mean_tortuosity": mean_tortuosity,
            "mean_large_vessel_tortuosity": mean_large_vessel_tortuosity,
            "mean_small_vessel_tortuosity": mean_small_vessel_tortuosity}


# def vessel_roundness_features(G: nx.Graph):
#     """Calculates the median roundness and median standard deviation of the vessels."""
#     median_roundness = np.median([data["roundnessAvg"] for _, _, data in G.edges(data=True)])
#     median_roundness_std = np.median([data["roundnessStd"] for _, _, data in G.edges(data=True)])
#     return {"median_roundness": median_roundness, "median_roundness_std": median_roundness_std}