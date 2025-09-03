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
    plot: bool = False,
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

    # make plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.log(1 / scales), np.log(np.unique(Ns)), c="teal", label="Measured ratios")
        ax.set_ylabel("$\log N(\epsilon)$")
        ax.set_xlabel("$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(coeffs, np.log(1 / scales))
        ax.plot(
            np.log(1 / scales),
            fitted_y_vals,
            "k--",
            label=f"Fit: {np.round(coeffs[0], 3)}X+{coeffs[1]}",
        )
        ax.legend()
    return coeffs[0]


def path_length(G: nx.Graph):
    """
    Calculates the average path length of the vessel network.

    Args:
        G: NetworkX graph representing vessel network
    Returns:
        float: Average path length of the vessel network
    """
    # AVERAGE PATH LENGTH IN MM
    path_lengths = []
    for v in G.nodes():
        spl = dict(nx.single_source_shortest_path_length(G, v))
        for tmp_path in spl:
            path_lengths.append(spl[tmp_path])

    avg_path_length = 0 if len(path_lengths) == 0 else (sum(path_lengths) / len(path_lengths))
    return avg_path_length


def median_bifurcation_exponent(G: nx.Graph):
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
    for edges in node_edges:
        radii = [e[2]["radius_avg"] for e in edges]
        radii.sort()
        parent = radii.pop()
        children = np.asarray(radii)

        def murray(x):
            a = np.sum(children**x)
            b = a ** (1 / x)
            c = parent - b
            d = np.abs(c)
            return d
            # return np.abs(parent - np.sum(children ** x) ** (1 / x))

        try:
            bif = minimize_scalar(murray, bounds=(1.0, 100.0), method="Bounded")
            all_bif_exp.append(bif.x)
        except:
            continue
    return np.nanmedian(all_bif_exp).item()


def largest_voxel_component_size(volume: np.ndarray):
    """Calculates the size of the largest connected component in a binary volume."""
    footprint = ndi.generate_binary_structure(volume.ndim, 1)
    ccs = np.zeros_like(volume, dtype=np.int32)
    ndi.label(volume, footprint, output=ccs)
    component_sizes = np.bincount(ccs.ravel())
    return (
        0 if len(component_sizes) < 2 else component_sizes[component_sizes.argsort()[-2]]
    )  # use -2, because -1 is the background


def cycle_features(G: nx.Graph):
    """Calculates the number of cycles, median cycle length and maximum cycle length of a graph."""
    # Convert multigraph to simple graph by keeping only one edge between nodes
    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    basis = nx.cycle_basis(G)
    return {
        "#cycles": len(basis),
        "median_cycle_length": 0 if len(basis) == 0 else np.median([len(b) for b in basis]),
        "max_cycle_ength": 0 if len(basis) == 0 else max([len(b) for b in basis]),
    }


def component_length_features(G: nx.Graph):
    """Calculates the median length per connected component, median distance per
    connected component and the length of the longest connected component."""
    lengths = []
    # distances = []
    for c in nx.connected_components(G):
        g = G.subgraph(c)
        # distances.append(sum([(data['distance']) for _, _, data in g.edges(data=True)]))
        lengths.append(sum([(data["length"]) for _, _, data in g.edges(data=True)]))
    return {
        "median_length_per_component": np.median(lengths),
        "longest_connected_component": np.max(lengths),
    }


def vessel_length_features(G: nx.Graph, large_vessel_radius: float):
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
    vessel_num = len(vessels)
    total_small_vessel_length = sum(small_vessels)  # max RadiusAvg is probably the most accurate
    small_vessel_num = len(small_vessels)
    total_large_vessel_length = sum(large_vessels)
    large_vessel_num = len(large_vessels)

    median_vessel_length = 0 if len(vessels) == 0 else np.median(vessels).item()
    median_large_vessel_length = 0 if len(large_vessels) == 0 else np.median(large_vessels).item()
    median_small_vessel_length = 0 if len(small_vessels) == 0 else np.median(small_vessels).item()

    return {
        "total_vessel_length": total_vessel_length,
        "vessel_num": vessel_num,
        "total_large_vessel_length": total_large_vessel_length,
        "large_vessel_num": large_vessel_num,
        "total_small_vessel_length": total_small_vessel_length,
        "small_vessel_num": small_vessel_num,
        "median_vessel_length": median_vessel_length,
        "median_large_vessel_length": median_large_vessel_length,
        "median_small_vessel_length": median_small_vessel_length,
    }


def blood_volume_features(volume: np.ndarray, volume_unfiltered: np.ndarray, spacing: Tuple):
    """Calculates the total and average blood volume, as well as the total blood volume of the unfiltered segmentation and the
    difference to the filtered one."""
    total_blood_volume = np.sum(volume) * np.prod(spacing)
    avg_blood_volume = np.mean(volume) * np.prod(spacing)
    if volume_unfiltered is None:
        return {"total_blood_volume": total_blood_volume, "avg_blood_volume": avg_blood_volume}
    total_blood_volume_unfiltered = np.sum(volume_unfiltered) * np.prod(spacing)
    filtered_out_blood_volume = total_blood_volume_unfiltered - total_blood_volume
    return {
        "total_blood_volume": total_blood_volume,
        "avg_blood_volume": avg_blood_volume,
        "total_blood_volume_unfiltered": total_blood_volume_unfiltered,
        "filtered_out_blood_volume": filtered_out_blood_volume,
    }


def layer_thickness_features_old(vol_in: np.ndarray):

    def func(x, a, c):
        return a * np.power(x, 2) + c

    # mip over short axis
    vol = remove_small_objects(vol_in > 0, min_size=3000)
    # vol = vol_in
    vol_mip = np.max(vol, axis=1)
    i_top = np.argmax(vol_mip, axis=0)

    window = vol_mip.shape[1] / 4
    if window % 2 == 0:
        window -= 1

    i_top_smooth = savgol_filter(i_top, window_length=41, polyorder=2)
    x = np.arange(len(i_top))
    params_top, _ = curve_fit(func, x, i_top_smooth, maxfev=10000)

    vol_sum = np.sum(vol, axis=1)  # > 8
    vol_sum = vol_sum > vol_sum.max() * 0.2
    i_bot = vol_sum.shape[0] - np.argmax(vol_sum[::-1, :], axis=0)

    data_masked = np.where(i_bot == vol_sum.shape[0], np.nan, i_bot)
    nans, x = np.isnan(data_masked), lambda z: z.nonzero()[0]
    interpolator = interp1d(
        x(~nans), data_masked[~nans], bounds_error=False, fill_value="extrapolate"
    )
    data_interpolated = np.where(nans, interpolator(np.arange(len(data_masked))), data_masked)
    i_bot_smooth = savgol_filter(data_interpolated, window_length=41, polyorder=2)

    x = np.arange(len(i_bot))
    params_bot, _ = curve_fit(func, x, i_bot_smooth, maxfev=10000)

    # Prepare x-axis (indices)
    x = np.arange(len(i_top))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, i_top, label="i_top (data)", color="blue")
    plt.plot(x, func(x, *params_top), label="i_top (fit)", color="orange")
    plt.plot(x, i_top_smooth, label="smooth", color="green")
    plt.title("Exponential Fit for i_top")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, i_bot, label="i_bot (data)", color="green")
    plt.plot(x, i_bot_smooth, label="i_bot (smooth)", color="blue")
    plt.plot(x, func(x, *params_bot), label="i_bot (fit)", color="orange")
    plt.title("Exponential Fit for i_bot")
    plt.legend()

    plt.figure()
    plt.imshow(vol_mip)
    plt.plot(x, func(x, *params_top), label="i_top (fit)", color="white")

    plt.figure()
    plt.imshow(vol_sum)
    plt.plot(x, func(x, *params_bot), label="i_bot (fit)", color="white")

    width = i_bot_smooth - i_top_smooth

    return np.mean(width)


def layer_thickness_features(vol_in: np.ndarray):

    def func(x, a, c):
        return a * np.power(x, 2) + c

    # mip over short axis
    vol = remove_small_objects(vol_in > 0, min_size=3000)
    # vol = vol_in
    vol_mip = np.max(vol, axis=1)
    if vol_mip.sum()/np.prod(vol_mip.shape) < 0.025:
        return 10 # Layer thickness evaluation works only when there are enough vessels present
    
    i_top = np.argmax(vol_mip, axis=0)

    window = vol_mip.shape[1] / 4
    if window % 2 == 0:
        window -= 1

    i_top_smooth = savgol_filter(i_top, window_length=41, polyorder=2)
    x = np.arange(len(i_top))
    params_top, _ = curve_fit(func, x, i_top_smooth, maxfev=10000)

    # # Define a 1D kernel for summing a window of 11 pixels along dim1
    # kernel = np.ones((1, 22))  # 1 row, 11 columns

    # # Convolve the kernel along dim1 (second axis)
    # vol_mip_sum = convolve2d(vol_mip, kernel, mode='same', boundary='wrap')>0

    pad_width = 20

    structure = ndimage.generate_binary_structure(2, 2)

    seg_vol = np.pad(vol_mip, pad_width=pad_width, mode="edge")
    seg_vol = ndimage.binary_closing(seg_vol, structure=structure, iterations=10, border_value=0)
    seg_vol = ndimage.binary_opening(seg_vol, structure=structure, iterations=10, border_value=0)
    # seg_vol = ndimage.binary_closing(seg_vol, structure=structure, iterations=5, border_value=0
    # )
    vol_sum = seg_vol[pad_width:-pad_width, pad_width:-pad_width]

    i_bot = vol_sum.shape[0] - np.argmax(vol_sum[::-1, :], axis=0)

    data_masked = np.where(i_bot == vol_sum.shape[0], np.nan, i_bot)

    nans, x = np.isnan(data_masked), lambda z: z.nonzero()[0]

    if nans.sum() > vol_sum.shape[1] * .7:
        return 10 # If more than 70% of mip needs to be interpolated

    interpolator = interp1d(
        x(~nans), data_masked[~nans], bounds_error=False, fill_value="extrapolate"
    )
    data_interpolated = np.where(nans, interpolator(np.arange(len(data_masked))), data_masked)
    i_bot_smooth = savgol_filter(data_interpolated, window_length=41, polyorder=2)

    x = np.arange(len(i_bot))
    params_bot, _ = curve_fit(func, x, i_bot_smooth, maxfev=10000)

    # Calculate area between curves
    def avg_width_between_curves(xs, params_top, params_bot):
        # Integrate the difference of the two quadratic functions
        a_top, c_top = params_top
        a_bot, c_bot = params_bot

        def fun(x):
            return (a_bot - a_top) * (x**3) / 3 + (c_bot - c_top) * x

        return (fun(xs[-1]) - fun(xs[0])) / len(xs)

    x = np.arange(len(i_top))
    width = avg_width_between_curves(x, params_top, params_bot)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(x, i_top, label="i_top (data)", color="blue")
    # plt.plot(x, func(x, *params_top), label="i_top (fit)", color="orange")
    # plt.plot(x, i_top_smooth, label="smooth", color="green")
    # plt.title("Exponential Fit for i_top")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(x, i_bot, label="i_bot (data)", color="green")
    # plt.plot(x, i_bot_smooth, label="i_bot (smooth)", color="blue")
    # plt.plot(x, func(x, *params_bot), label="i_bot (fit)", color="orange")
    # plt.title("Exponential Fit for i_bot")
    # plt.legend()

    # plt.figure()
    # plt.imshow(vol_mip)
    # plt.plot(x, func(x, *params_top), label="i_top (fit)", color="white")
    # plt.plot(x, func(x, *params_bot), label="i_bot (fit)", color="white")

    # plt.figure()
    # plt.imshow(vol_sum)
    # plt.plot(x, func(x, *params_bot), label="i_bot (fit)", color="white")
    # plt.show()

    return width


def bifurcation_features(G: nx.Graph, image_volume: float,vol_filtered:np.ndarray):
    """Calculates the number of bifurcations (in total and normalized by the image volume."""
    if vol_filtered.sum() == 0:
        return {
            "#bifurcations": 0,
            "#bifurcations_normalized": 0,
            "nikoletta_branch_number": 0,
            "nikoletta_branch_j2e_total": 0,
            "nikoletta_branch_j2j_total": 0,
            "nikoletta_num_junctions": 0,
        } 
    n_bifurcations = sum(1 for node, degree in G.degree() if degree > 2)

    info = skan.summarize(skan.Skeleton(skeletonize(vol_filtered, method='lee')))
    branch_data = info.loc[info['branch-distance'] > 9]
    
    nikoletta_branch_number =  len(branch_data['branch-distance'].values)
    nikoletta_branch_j2e_total = np.sum(branch_data['branch-type'].values == 1)
    nikoletta_branch_j2j_total = np.sum(branch_data['branch-type'].values == 2)
    nikoletta_num_junctions = np.unique(branch_data['node-id-src'].values).shape[0]

    return {
        "#bifurcations": n_bifurcations,
        "#bifurcations_normalized": n_bifurcations / image_volume,
        "nikoletta_branch_number": nikoletta_branch_number,
        "nikoletta_branch_j2e_total": nikoletta_branch_j2e_total,
        "nikoletta_branch_j2j_total": nikoletta_branch_j2j_total,
        "nikoletta_num_junctions": nikoletta_num_junctions,
    }


def radius_features(G: nx.Graph):
    median_radius = np.median([data["radius_avg"] for _, _, data in G.edges(data=True)])
    median_radius_std = np.median([data["radius_SD"] for _, _, data in G.edges(data=True)])
    median_variationCoeff = median_radius_std / median_radius
    return {
        "median_radius": median_radius,
        "median_radius_std": median_radius_std,
        "median_variationCoeff": median_variationCoeff,
    }


def graph_metric_features(G: nx.Graph):
    density = nx.density(G)
    degree_assortativity_coefficient = (
        0
        if nx.degree_assortativity_coefficient(G) is None
        else nx.degree_assortativity_coefficient(G)
    )
    avg_degree = (
        0
        if G.number_of_nodes() == 0
        else sum(degree for node, degree in G.degree()) / G.number_of_nodes()
    )
    n_components = len(list(nx.connected_components(G)))
    return {
        "density": density,
        "degree_assortativity_coefficient": degree_assortativity_coefficient,
        "avg_degree": avg_degree,
        "#components": n_components,
    }


def vessel_curveness_features(G: nx.Graph):
    """Calculates the median curveness and straightness of the vessels."""
    median_curveness = np.median([data["curveness"] for _, _, data in G.edges(data=True)])
    median_straightness = np.median(
        [(data["distance"] / data["length"]) for _, _, data in G.edges(data=True)]
    )
    return {"median_curveness": median_curveness, "median_straightness": median_straightness}


def vessel_tortuosity_features(G: nx.Graph):
    """Calculates the median tortuosity of the vessels."""
    median_tortuosity = np.median([data["tortuosity"] for _, _, data in G.edges(data=True)])
    return {"median_tortuosity": median_tortuosity}


def vessel_roundness_features(G: nx.Graph):
    """Calculates the median roundness and median standard deviation of the vessels."""
    median_roundness = np.median([data["roundnessAvg"] for _, _, data in G.edges(data=True)])
    median_roundness_std = np.median([data["roundnessStd"] for _, _, data in G.edges(data=True)])
    return {"median_roundness": median_roundness, "median_roundness_std": median_roundness_std}
