import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd

from vvl.utils.GraphInfo import GraphInfo

vesselseg_dir = r"D:\data\tmp\delme\proc\vessel_segmentation"
layerseg_dir = r"D:\data\tmp\delme\proc\epidermis_segmentation"

filter_length = 0.250  # remove paths with a length less than this
prune_length = 0.0  # remove connected endpoint vessels with length less than this
# large_vessel_radius = None  # Manually define at which radius vessels are considered large
large_vessel_radius = 0.020991214602581708 # Old reference value: 0.015997390253347205
# Tartu Baselines: 0.020991214602581708
# German Cohort Baselines: 0.01832396790159651
vp_depth = 40  # Depth at which to seperate the vessels into upper and lower region
legacy = True
normalize = False # If vessel signal is already cropped to normalized volume then don't need to normalize

resolution = [0.012, 0.012, 0.003] 



def find_layseg_for_vesseg(vesseg_path, layseg_dir):
    def clean_up_name(name):
        name = name.replace("_processed", "")
        name = name.replace("__l", "")
        name = name.replace("_l", "")
        name = name.replace("_0000", "")
        name = name.replace("_0001", "")
        name = name.replace("_ed.nii.gz", "")
        name = name.replace("_ves.nii.gz", "")
        name = name.replace(".nii.gz", "")
        return name

    lays = os.listdir(layseg_dir)
    for lay in lays:
        if clean_up_name(lay) == clean_up_name(os.path.basename(vesseg_path)):
            return os.path.join(layseg_dir, lay)
    raise ValueError(f"No matching layseg found for {vesseg_path} in {layseg_dir}")


def extract_graph_wrapper(g_i):
    g_i.extract_graph()
    return g_i


def extract_radii_wrapper(g_i):
    return g_i.extract_radius()

def extract_feats_wrapper(g_i):
    g_i.extract_features()
    return g_i

def extract_sizefeats_wrapper(g_i):
    # g_i.extract_features()
    g_i.extract_features_upper_lower()

    return g_i.features


if __name__ == "__main__":
    results_folder = os.path.join(vesselseg_dir, "vesselvio")
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    graph_infos = []
    # Load all files
    print("Loading files...")
    segs = [seg for seg in os.listdir(vesselseg_dir) if ".nii.gz" in seg]
    for vesselseg_name in tqdm(segs):
        vesselseg_path = os.path.join(vesselseg_dir, vesselseg_name)
        graph_info = GraphInfo(
            vesselseg_path,
            find_layseg_for_vesseg(vesselseg_path, layerseg_dir),
            resolution=resolution,
            filter_length=filter_length,
            prune_length=prune_length,
            legacy=legacy,
            output_dir=results_folder,
            depth = vp_depth,
            normalize=normalize,
        )
        graph_infos.append(graph_info)

    # Extract Graphs
    print("Extracting graphs...")
    with Pool() as pool:
        for i, result in enumerate(tqdm(pool.imap(extract_graph_wrapper, graph_infos), total=len(graph_infos))):
            graph_infos[i] = result

    # # Extract radii
    # print("Extracting radii...")
    # with Pool() as pool:
    #     radii = list(tqdm(pool.imap(extract_radii_wrapper, graph_infos), total=len(graph_infos)))

    # large_vessel_radius = np.median(radii)

    print(f"Large vessel radius is {large_vessel_radius}")
    for g_i in graph_infos:
        g_i.large_vessel_radius = large_vessel_radius

    print(large_vessel_radius)
    
    # Extract size-dependent features
    print("Extracting size-dependent  features...")
    with Pool(maxtasksperchild=32) as pool:
        for i, result in enumerate(tqdm(pool.imap(extract_sizefeats_wrapper, graph_infos), total=len(graph_infos))):
            graph_infos[i].features = result

    # for g_i in tqdm(graph_infos):
    #     g_i.features = g_i.extract_features_upper_lower()


    feature_list = [g_i.features for g_i in graph_infos]
    df = pd.DataFrame(feature_list)
    df.to_csv(os.path.join(results_folder, "features.csv"), index=False)
    df.to_excel(os.path.join(results_folder, "features.xlsx"), index=False)
