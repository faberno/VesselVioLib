from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import time
from typing import Optional

import igraph as ig
from tqdm import tqdm

from vvl import logger
from vvl.volume_processing import prepare_volume, skeletonize_volume, radii_calc_input
from vvl.input_classes import AnalysisOptions, AnnotationOptions, VisualizationOptions
from vvl.image_processing import load_volume, reshape_2D
from vvl.graph_processing import create_graph, prune_input, filter_input
from vvl.feature_extraction import feature_input
from vvl.results_export import cache_result, write_seg_results
from vvl.graph_io import save_graph

# try:
#     from . import volume_visualization as VolVis
# except ImportError:
#     print("Could not import volume_visualization.")


from vvl.annotation import segmentation, segmentation_prep, labeling, tree_processing


# Process raw segmented volumes
def process_volume(volume_file: str,
                   analysis_options: AnalysisOptions,
                   annotation_options: AnnotationOptions=None,
                   visualization_options: VisualizationOptions=None):

    tic = time.perf_counter()

    volume_file = Path(volume_file)
    assert volume_file.is_file()

    filename = volume_file.stem
    logger.info(f"Processing dataset: {filename}")


    if not (annotation_options is None
            or annotation_options.annotation_type == "None"
            or annotation_options.annotation_type is None):
        raise NotImplementedError("VesselVio's annotation processing feature is "
                                  "currently not yet supported in VesselVioLib.")

    #if visualization_options is None:
        #visualization_options = IC.VisualizationOptions()


    # Image and volume processing.
    volume = load_volume(
        str(volume_file),
        allow_image_binarization=analysis_options.allow_image_binarization)
    volume, volume_crop_start = prepare_volume(volume)
    skeleton_pointcloud = skeletonize_volume(volume)

    # Calculate radii
    skeleton_radii = radii_calc_input(
        volume,
        skeleton_pointcloud,
        analysis_options.resolution
    )

    # Now, we can treat 2D arrays as 3D arrays for compatibility
    # with the rest of our pipeline.
    if volume.ndim == 2:
        skeleton_pointcloud, volume, volume_shape = reshape_2D(
            skeleton_pointcloud, volume
        )

    volume_shape = volume.shape
    del volume

    graph = create_graph(
        volume_shape,
        skeleton_radii,
        skeleton_pointcloud,
        volume_crop_start,
    )

    if analysis_options.prune_length > 0:
        # Prune connected endpoint segments based on a user-defined length
        prune_input(
            graph,
            analysis_options.prune_length,
            analysis_options
        )

    # Filter isolated segments that are shorter than defined length
    # If visualizing the dataset, filter these from the volume as well.
    filter_input(
        graph,
        analysis_options.filter_length,
        analysis_options.resolution
    )

    ## Analysis.
    result, seg_results = feature_input(
        graph,
        analysis_options.resolution,
        filename,
        image_dim=analysis_options.image_dimensions,
        image_shape=volume_shape,
        save_seg_results=analysis_options.save_seg_results,
        reduce_graph=analysis_options.save_graph,
    )

    cache_result(result)  # Cache results

    if analysis_options.save_seg_results:
        write_seg_results(seg_results, results_folder, filename, roi_name)

    if analysis_options.save_graph:
        save_graph(graph, filename, results_folder, verbose=verbose)

    graph.es["hex"] = ["FFFFFF"]