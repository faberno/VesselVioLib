import logging
logger = logging.getLogger(__name__)

from math import inf

import igraph as ig
from tqdm import tqdm

# from . import (graph_io as GIO,
#              results_export as ResExp,
#              volume_processing as VolProc,
#              helpers,
#              input_classes as IC,
#              graph_processing as GProc,
#              image_processing as ImProc,
#              feature_extraction as FeatExt)

from vvl.input_classes import AnalysisOptions, VisualizationOptions
from vvl.image_processing import load_volume
from vvl.volume_processing import skeletonize, calculate_centerline_radii
from vvl.graph_processing import create_graph, filter_graph_edges

def extract_graph_from_volume(
        volume_file: str,
        minimum_endpoint_segment_length: float = 0.0,
        minimum_isolated_segment_length: float = inf):
    """
    # DOCTODO #

    Args:
        volume_file:
        minimum_endpoint_segment_length: prune endpoint segments shorter than this length
        minimum_isolated_segment_length: filter isolated segments shorter than this length

    Returns:

    """

    volume, resolution = load_volume(volume_file)  # todo: raise error if failed or not binary, add enforce binary option
    # volume, point_minima = VolProc.volume_prep(volume)
    # volume = pad_volume(volume, analysis_options.padding)
    skeleton, centerlines = skeletonize(volume)

    centerline_radii = calculate_centerline_radii(
        volume,
        centerlines,
        resolution
    )

    # todo: handle 2d

    graph = create_graph(
        centerlines,
        centerline_radii,
        volume.shape
    )

    filter_graph_edges(
        graph,
        minimum_endpoint_segment_length,
        minimum_isolated_segment_length,
        resolution
    )

    #
    # # Filter isolated segments that are shorter than defined length
    # # If visualizing the dataset, filter these from the volume as well.
    # filter_input(graph, gen_options.filter_length, resolution, verbose=verbose)


