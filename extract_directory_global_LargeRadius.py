import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import nibabel as nib
import pandas as pd

from vvl.utils.GraphInfo import GraphInfo

seg_dir = r"D:\data\joshua\Atlas\vessel_preds\vp\preds038"

resolution = [3, 12, 12]
filter_length = 250
prune_length = 0.0
vol_spacing = np.array([0.003, 0.012, 0.012])
large_vessel_radius = 14.4


def extract_graph_wrapper(g_i):
    g_i.extract_graph()
    return g_i


def extract_radii_wrapper(g_i):
    return g_i.extract_radius()


def extract_sizefeats_wrapper(g_i):
    g_i.extract_features()
    return g_i


if __name__ == "__main__":
    results_folder = os.path.join(seg_dir, "vesselvio")
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    graph_infos = []
    # Load all files
    print("Loading files...")
    segs = [seg for seg in os.listdir(seg_dir) if ".nii.gz" in seg]
    for volume_name in tqdm(segs):
        volume_path = os.path.join(seg_dir, volume_name)
        graph_info = GraphInfo(
            volume_path,
            resolution=resolution,
            vol_spacing=vol_spacing,
            filter_length=filter_length,
            prune_length=prune_length,
            output_dir=results_folder,
        )
        graph_infos.append(graph_info)

    # Extract Graphs
    print("Extracting graphs...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(extract_graph_wrapper, graph_infos), total=len(graph_infos)))
    graph_infos.clear()
    graph_infos.extend(results)

    for g_i in graph_infos:
        g_i.large_vessel_radius = large_vessel_radius

    # Extract size-dependent features
    print("Extracting size-dependent  features...")
    with Pool() as pool:
        results = list(
            tqdm(pool.imap(extract_sizefeats_wrapper, graph_infos), total=len(graph_infos))
        )
    graph_infos.clear()
    graph_infos.extend(results)

    feature_list = [g_i.features for g_i in graph_infos]
    df = pd.DataFrame(feature_list)
    df.to_csv(os.path.join(results_folder, "features.csv"), index=False)
