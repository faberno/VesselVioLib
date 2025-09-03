import networkx as nx
import re
import nibabel as nib
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects
import os
from multiprocessing import Pool

from functools import partial
from vvl.feature_extraction_more import (
    bifurcation_features,
    blood_volume_features,
    component_length_features,
    cycle_features,
    fractal_dimension,
    graph_metric_features,
    largest_voxel_component_size,
    median_bifurcation_exponent,
    vessel_length_features,
    vessel_tortuosity_features,
    path_length,
    radius_features,
    layer_thickness_features,
)

"""
Features we have in voreen but not in vesselvio:

distance   -> Dont care
curveness   -> We do have tortuosity
avgCrossSection -> Not necessary when we have radius_avg
roundness -> Measure of maximum vs minimum radius, More of a measure of segmentation quality than real biomedical feature
node1_degree -> Not required since we have the degree from networkx
node2_degree -> Not required since we have the degree from networkx
num_voxels -> Dont care
"""


def extract_timestamp(filename: str):
    try:
        return re.search(r"\d{14}", filename).group()
    except AttributeError:
        filename = filename.replace(".graphml", "")
        filename = filename.replace(".nii.gz", "")
        filename = filename.replace(".nii", "")
        filename = filename.replace("_processed", "")
        return filename


def extract_features(
    G, vol, vol_filtered, large_vessel_radius=21.0, vol_spacing=np.array([0.003, 0.012, 0.012])
):
    """Extract vessel network features from graph and volume data.

    Args:
        G: NetworkX graph representing vessel network
        vol: Original volume data
        vol_filtered: Filtered volume data
        large_vessel_radius: Threshold for large vessel classification (default: 4.0)
        vol_spacing: Voxel spacing in mm (default: [0.003, 0.012, 0.012])

    Returns:
        dict: Dictionary containing extracted vessel features
    """
    voxel_volume = np.prod(vol_spacing)  # volume of a single voxel
    total_volume = np.prod(vol_filtered.shape) * voxel_volume

    features = {}

    features["avg_path_length"] = path_length(G)

    features["medBifExponent"] = median_bifurcation_exponent(G)
    features.update(vessel_length_features(G, large_vessel_radius))
    features.update(bifurcation_features(G, total_volume, vol_filtered))
    features.update(blood_volume_features(vol_filtered, vol, vol_spacing))
    features.update(graph_metric_features(G))

    features.update(component_length_features(G))
    features.update(cycle_features(G))
    features.update(radius_features(G))
    features.update(vessel_tortuosity_features(G))
    # features.update(vessel_roundness_features(G)) # No roundness
    # features.update(vessel_curveness_features(G))

    features["volume_largest_component"] = (
        largest_voxel_component_size(vol_filtered) * vol_spacing.prod()
    )
    features["fractal_dimension"] = fractal_dimension(vol_filtered)

    features["layer_thickness"] = layer_thickness_features(vol)

    return features


def process_single_graph(graph, graphs_dir, seg_dir):
    """Process a single graph file and return its features"""
    timestamp = extract_timestamp(graph)
    seg_filename = [f for f in os.listdir(seg_dir) if timestamp in f][0]

    g = nx.read_graphml(os.path.join(graphs_dir, graph))
    seg = os.path.join(seg_dir, seg_filename)

    vol = np.asanyarray(nib.load(seg).dataobj)
    vol_filtered = remove_small_objects(vol > 0, min_size=500)

    features = extract_features(g, vol, vol_filtered)
    features["name"] = graph.replace(".graphml", "")
    return features


GRAPHS = r"GRAPHS/PATH"
SEGMENTATIONS = r"SEGMENTATIONS/PATH"

if __name__ == "__main__":

    feature_list = []
    graphs = [f for f in os.listdir(GRAPHS) if f.endswith(".graphml")]

    # Create a partial function with the fixed arguments
    process_func = partial(process_single_graph, graphs_dir=GRAPHS, seg_dir=SEGMENTATIONS)
    # num_cores = cpu_count()  # Or specify a number like cpu_count() - 1

    # with ThreadPoolExecutor(max_workers=num_cores) as executor:
    #     results = list(tqdm(executor.map(process_func, graphs), total=len(graphs)))

    with Pool() as p:
        results = list(tqdm(p.imap(process_func, graphs), total=len(graphs)))

    # results = []
    # for g in tqdm(graphs):
    #     results.append(process_func(g))

    feature_list.extend(results)


    df = pd.DataFrame(feature_list)
    df.to_csv(os.path.join(GRAPHS, "vesselvio_features.csv"))
