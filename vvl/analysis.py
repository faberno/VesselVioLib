import logging
from typing import Sequence
import os
import pandas as pd

from vvl.utils.image_processing import get_filename, prep_resolution, load_volume, binary_check, reshape_2D
from vvl.utils.volume_processing import volume_prep, pad_volume, skeletonize, radii_calc_input
from vvl.utils.graph_processing import create_graph, prune_input, filter_input
from vvl.utils.feature_processing import feature_input, process_single_graph
from vvl.utils.io import save_graph

logger = logging.getLogger(__name__)

import numpy as np
from scipy.ndimage import distance_transform_edt



def assign_centerline_indices(volume, centerline_mask, centerline_voxels, spacing):
    shape = centerline_mask.shape
    centerline_mask = centerline_mask.astype(bool)

    # Compute distance transform on the inverse centerline
    # `indices` will contain for each voxel the coordinate of its nearest centerline voxel
    distances, indices = distance_transform_edt(~centerline_mask, return_indices=True, sampling=spacing)

    # Map each coordinate to the corresponding index in centerline_voxels
    # First, make a lookup dictionary from coordinates to centerline indices
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(centerline_voxels)}

    # Get the nearest centerline coordinates for all voxels
    z_idx, y_idx, x_idx = indices
    nearest_coords = np.stack([z_idx, y_idx, x_idx], axis=-1)

    # Prepare the output volume
    output = np.full(shape, fill_value=-1, dtype=int)

    # Only assign for segmented voxels
    for coord in zip(*np.where(volume)):
        nearest = tuple(nearest_coords[coord])
        output[coord] = coord_to_idx.get(nearest, -1)  # -1 if somehow not found

    return output

def reconstruct_volume(volume, graph, points, resolution):
    def get_mask_from_subset(A, B):
        # Convert A to a structured array for fast matching
        A_struct = A.view([('', A.dtype)] * A.shape[1])
        B_struct = B.view([('', B.dtype)] * B.shape[1])

        # Use np.isin to find which rows of A are in B
        mask = np.isin(A_struct, B_struct)
        return mask[:, 0]

    centerline_mask = np.zeros(volume.shape, dtype=bool)
    centerline_mask[tuple(points.T)] = True

    # Compute distance transform on the inverse centerline
    # `indices` will contain for each voxel the coordinate of its nearest centerline voxel
    distances, indices = distance_transform_edt(~centerline_mask, return_indices=True, sampling=resolution)

    # Map each coordinate to the corresponding index in centerline_voxels
    # First, make a lookup dictionary from coordinates to centerline indices
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(points)}

    # Get the nearest centerline coordinates for all voxels
    z_idx, y_idx, x_idx = indices
    nearest_coords = np.stack([z_idx, y_idx, x_idx], axis=-1)

    # Prepare the output volume
    assign_volume = np.full(volume.shape, fill_value=-1, dtype=int)

    # Only assign for segmented voxels
    for coord in zip(*np.where(volume)):
        nearest = tuple(nearest_coords[coord])
        assign_volume[coord] = coord_to_idx.get(nearest, -1)  # -1 if somehow not found

    filtered_points = np.array([v['v_coords'] for v in graph.vs])
    filter_mask = get_mask_from_subset(points, filtered_points)
    filter_indices = np.where(~filter_mask)[0]
    volume_mask = np.isin(assign_volume, filter_indices)
    assign_volume[volume_mask] = -1

    return assign_volume


def extract_graph_from_volume(
        volume_file: str,
        resolution: Sequence,
        filter_length: float,
        prune_length: float = 0.0):
    filename = get_filename(volume_file)

    resolution = prep_resolution(resolution)

    volume, image_shape = load_volume(volume_file)

    if volume is None:
        raise ValueError(f"Failed to load volume from {volume_file}. Please check the file path and format.")
    elif not binary_check(volume):
        raise ValueError(
            f"The volume data in {volume_file} is not binary. Please ensure the data is binary (0s and 1s).")

    volume, point_minima = volume_prep(volume)

    volume = pad_volume(volume)

    points = skeletonize(volume)

    skeleton_radii, vis_radii = radii_calc_input(
        volume,
        points,
        resolution,
        gen_vis_radii=True
    )

    if volume.ndim == 2:
        points, volume, volume_shape = reshape_2D(points, volume)
    else:
        volume_shape = volume.shape

    graph = create_graph(
        volume_shape,
        skeleton_radii,
        vis_radii,
        points,
        point_minima,
    )

    if prune_length > 0:
        # Prune connected endpoint segments based on a user-defined length
        prune_input(graph, prune_length, resolution)

    # Filter isolated segments that are shorter than defined length
    # If visualizing the dataset, filter these from the volume as well.
    filter_input(graph, filter_length, resolution)

    filtered_volume = reconstruct_volume(volume, graph, points, resolution) >= 0

    # endregion
    ## Analysis.
    result, seg_results = feature_input(
        graph,
        resolution,
        filename,
        image_dim=volume.ndim,
        image_shape=image_shape,
        roi_name="None",
        roi_volume="NA",
        save_seg_results=False,
        # Reduce graph if saving or visualizing
        reduce_graph=True
    )

    return graph, filtered_volume

    # results_folder = r'C:\Users\fabia\PycharmProjects\VesselVioLib\data\vesselvio'
    # save_graph(graph, filename, results_folder)
    #
    # # --------------
    #
    # seg_dir = r'C:\Users\fabia\PycharmProjects\VesselVioLib\data'
    # graph_dir = os.path.join(seg_dir, "vesselvio/Graphs")
    # feature_list = []
    # graphs = [f for f in os.listdir(graph_dir) if f.endswith(".graphml")]
    #
    # for graph in graphs:
    #     results = process_single_graph(graph, graphs_dir=graph_dir, seg_dir=seg_dir)
    #     # results = []
    #     # for g in tqdm(graphs):
    #     #     results.append(process_func(g))
    #
    #     feature_list.append(results)
    #
    # df = pd.DataFrame(feature_list)
    # df.to_csv(os.path.join(graph_dir, "vesselvio_features.csv"))


def extract_graph_and_volume_features(graph, volume):
    print()