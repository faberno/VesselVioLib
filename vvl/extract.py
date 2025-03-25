from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import time

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


    if annotation_options is None:
        annotation_data = None
    else:
        annotation_data = tree_processing.convert_annotation_data(
            annotation_options.annotation_regions,
            annotation_options.annotation_atlas
        )
        roi_array = segmentation_prep.build_roi_array(
            annotation_data,
            annotation_type=annotation_options.annotation_type
        )

    volume, image_shape = ImProc.load_volume(str(volume_file), verbose=verbose)

    if annotation_data is None:
        volume, point_minima = VolProc.volume_prep(volume)
        roi_name = "None"
        roi_volume = "NA"
    else:
        raise NotImplementedError()

        g_main = ig.Graph()

        for i, roi_name in enumerate(annotation_data.keys()):
            if verbose and roi_name:
                if roi_name:
                    print(f"Analyzing {filename}: {roi_name}.")
                else:
                    print(f"Analyzing {filename}.")

            ## Image and volume processing.
            # region
            volume, image_shape = ImProc.load_volume(volume_file, verbose=verbose)

            if volume is None:  # make sure the image was loaded.
                if verbose:
                    print("Error loading volume.")
                break
            elif not ImProc.binary_check(volume):
                if verbose:
                    print("Error: Non-binary image loaded.")
                break

            # If there as an ROI, segment the ROI from the volume.
            if roi_name:
                roi_id = i % 255
                if roi_id == 0:
                    if not helpers.check_storage(volume_file):
                        file_size = helpers.get_file_size(volume_file, GB=True)
                        if verbose:
                            print(
                                f"Not enough disk space! Need at least {file_size:.1f}GB of free space for the volume annotation."
                            )
                        return

                    # We have to relabel every 255 elements because the volume.dtype == uint8.
                    roi_sub_array = roi_array[i : i + 255]
                    roi_volumes, minima, maxima = labeling.volume_labeling_input(
                        volume,
                        ann_options.annotation_file,
                        roi_sub_array,
                        ann_options.annotation_type,
                        verbose=verbose,
                    )
                    if roi_volumes is None:
                        break
                roi_volume = roi_volumes[roi_id]
                if roi_volume > 0:
                    point_minima, point_maxima = minima[roi_id], maxima[roi_id]
                    volume = segmentation.segmentation_input(
                        point_minima, point_maxima, roi_id + 1, verbose=verbose
                    )

                # Make sure the ROI is in the volume.
                if not roi_volume or not ImProc.segmentation_check(volume):
                    ResExp.cache_result([filename, roi_name, "ROI not in dataset."])  # Cache results
                    if verbose:
                        print("ROI Not in dataset.")
                    continue

        # Pad the volume for skeletonizatino
        volume = VolProc.pad_volume(volume)

        # Skeletonize, then find radii of skeleton points
        points = VolProc.skeletonize(volume, verbose=verbose)

        # Calculate radii
        skeleton_radii, vis_radii = VolProc.radii_calc_input(
            volume,
            points,
            resolution,
            gen_vis_radii=vis_options.visualize or gen_options.save_graph,
            verbose=verbose,
        )

        # Now, we can treat 2D arrays as 3D arrays for compatibility
        # with the rest of our pipeline.
        if volume.ndim == 2:
            points, volume, volume_shape = ImProc.reshape_2D(points, volume, verbose=verbose)
        else:
            volume_shape = volume.shape

        # At this point, delete the volume
        del volume
        # endregion

        ## Graph construction.
        # region
        # Send information to graph network creation.
        graph = GProc.create_graph(
            volume_shape,
            skeleton_radii,
            vis_radii,
            points,
            point_minima,
            verbose=verbose,
        )

        if gen_options.prune_length > 0:
            # Prune connected endpoint segments based on a user-defined length
            GProc.prune_input(graph, gen_options.prune_length, resolution, verbose=verbose)

        # Filter isolated segments that are shorter than defined length
        # If visualizing the dataset, filter these from the volume as well.
        GProc.filter_input(graph, gen_options.filter_length, resolution, verbose=verbose)

        # endregion
        ## Analysis.
        result, seg_results = FeatExt.feature_input(
            graph,
            resolution,
            filename,
            image_dim=gen_options.image_dimensions,
            image_shape=image_shape,
            roi_name=roi_name,
            roi_volume=roi_volume,
            save_seg_results=gen_options.save_seg_results,
            # Reduce graph if saving or visualizing
            reduce_graph=vis_options.visualize or gen_options.save_graph,
            verbose=verbose,
        )
        ResExp.cache_result(result)  # Cache results

        if gen_options.save_seg_results:
            ResExp.write_seg_results(seg_results, gen_options.results_folder, filename, roi_name)

        if gen_options.save_graph and not vis_options.visualize:
            GIO.save_graph(graph, filename, gen_options.results_folder, verbose=verbose)

        if roi_name != "None":
            graph.es["hex"] = [annotation_data[roi_name]["colors"][0]]
            graph.es["roi_ID"] = i
        else:
            graph.es["hex"] = ["FFFFFF"]
        g_main += graph
        del graph

    if verbose:
        print(
            f"Dataset analysis completed in a total "
            f"of {time.perf_counter() - tic:0.2f} seconds."
        )

    # Pad the volume for skeletonizatino
    volume = VolProc.pad_volume(volume)

    # Skeletonize, then find radii of skeleton points
    points = VolProc.skeletonize(volume, verbose=verbose)

    # Calculate radii
    skeleton_radii, vis_radii = VolProc.radii_calc_input(
        volume,
        points,
        resolution,
        gen_vis_radii=vis_options.visualize or gen_options.save_graph,
        verbose=verbose,
    )

    ## Visualization
    if vis_options.visualize:
        if (
            not vis_options.visualize
            or not vis_options.load_smoothed
            and not vis_options.load_original
        ):
            volume = None
        else:
            volume, _ = ImProc.load_volume(volume_file)
            volume = ImProc.prep_numba_compatability(volume)
            # Don't bound for visualization, as points will be true, not relative
            volume = VolProc.pad_volume(volume)
            if volume.ndim == 2:
                _, volume, _ = ImProc.reshape_2D(points, volume, verbose=verbose)

        VolVis.mesh_construction(g_main, vis_options, volume, iteration=iteration, verbose=verbose)

    # ResExp.write_results(gen_options.results_folder, gen_options.image_dimensions)

    # Make sure we delete the labeled_cache_volume if it exists
    ImProc.clear_labeled_cache()
    return