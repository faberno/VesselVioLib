from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import time
from typing import Optional

import igraph as ig
from tqdm import tqdm

from . import (graph_io as GIO,
             results_export as ResExp,
             volume_processing as VolProc,
             helpers,
             input_classes as IC,
             graph_processing as GProc,
             image_processing as ImProc,
             feature_extraction as FeatExt)

try:
    from . import volume_visualization as VolVis
except ImportError:
    print("Could not import volume_visualization.")


from .annotation import segmentation, segmentation_prep, labeling, tree_processing


# Process raw segmented volumes
def process_volume(volume_file: str,
                   analysis_options: IC.AnalysisOptions,
                   annotation_options: IC.AnnotationOptions=None,
                   visualization_options: IC.VisualizationOptions=None,
                   verbose=True):

    volume_file = Path(volume_file)
    assert volume_file.is_file()

    filename = volume_file.stem
    if verbose:
        tic = time.perf_counter()
        print("Processing dataset:", filename)


    if (annotation_options is None
            or annotation_options.annotation_type == "None"
            or annotation_options.annotation_type is None):
        annotation_data = {None: None}
    else:
        annotation_data = tree_processing.convert_annotation_data(
            annotation_options.annotation_regions,
            annotation_options.annotation_atlas
        )
        roi_array = segmentation_prep.build_roi_array(
            annotation_data,
            annotation_type=annotation_options.annotation_type
        )

    main_graph = ig.Graph()

    for i_roi, roi_name in enumerate(annotation_data.keys()):
        if verbose:
            if roi_name:
                print(f"Analyzing {filename}: {roi_name}.")
            else:
                print(f"Analyzing {filename}.")

        # Image and volume processing.
        volume = ImProc.load_volume(volume_file, verbose=verbose)
        if volume is None:  # make sure the image was loaded.
            if verbose:
                print("Error loading volume.")
            break
        elif not ImProc.binary_check(volume):
            if verbose:
                print("Error: Non-binary image loaded.")
            break
