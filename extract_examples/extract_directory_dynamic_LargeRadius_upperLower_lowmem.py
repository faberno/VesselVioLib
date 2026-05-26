import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd

from vvl.utils.GraphInfo import GraphInfo

vesselseg_dir = r"E:\CVD_backup\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\vessel\test"
layerseg_dir = r"E:\CVD_backup\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\epidermis"

filter_length = 0.250  # remove paths with a length less than this
prune_length = 0.0  # remove connected endpoint vessels with length less than this
# large_vessel_radius = None  # Manually define at which radius vessels are considered large
large_vessel_radius = 0.020991214602581708  # TARTU Baselines # Old reference value: 0.015997390253347205
vp_depth = 40  # Depth at which to separate the vessels into upper and lower region
legacy = True
normalize = False  # If vessel signal is already cropped to normalized volume then don't need to normalize


resolution = [0.012, 0.012, 0.003]


def find_layseg_for_vesseg(vesseg_path, layseg_dir):
    def clean_up_name(name):
        name = name.replace("_processed", "")
        name = name.replace("__l", "")
        name = name.replace("_l", "")
        name = name.replace("_0000", "")
        name = name.replace("_0001", "")
        name = name.replace("_ves.nii.gz", "")
        name = name.replace("_ed.nii.gz", "")
        name = name.replace(".nii.gz", "")
        return name

    lays = os.listdir(layseg_dir)
    for lay in lays:
        if clean_up_name(lay) == clean_up_name(os.path.basename(vesseg_path)):
            return os.path.join(layseg_dir, lay)
    raise ValueError(f"No matching layseg found for {vesseg_path} in {layseg_dir}")


def extract_radius_wrapper(args):
    vesselseg_path, layerseg_path, results_folder = args
    g_i = GraphInfo(
        vesselseg_path, layerseg_path,
        resolution=resolution, filter_length=filter_length,
        prune_length=prune_length, legacy=legacy,
        output_dir=results_folder, depth=vp_depth, normalize=normalize,
    )
    g_i.extract_graph()
    return g_i.extract_radius()


def process_single_file(args):
    vesselseg_path, layerseg_path, results_folder, large_vessel_radius = args
    g_i = GraphInfo(
        vesselseg_path, layerseg_path,
        resolution=resolution, filter_length=filter_length,
        prune_length=prune_length, legacy=legacy,
        output_dir=results_folder, depth=vp_depth, normalize=normalize,
    )
    g_i.large_vessel_radius = large_vessel_radius
    g_i.extract_graph()
    g_i.extract_features_upper_lower()
    return g_i.features


if __name__ == "__main__":
    results_folder = os.path.join(vesselseg_dir, "vesselvio")
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    segs = [seg for seg in os.listdir(vesselseg_dir) if ".nii.gz" in seg]
    seg_paths = [
        (
            os.path.join(vesselseg_dir, seg),
            find_layseg_for_vesseg(os.path.join(vesselseg_dir, seg), layerseg_dir),
            results_folder,
        )
        for seg in segs
    ]

    # Pass 1: extract graphs, compute dataset-wide median vessel radius
    print("Extracting radii...")
    with Pool(processes=2, maxtasksperchild=1) as pool:
        radii = list(tqdm(pool.imap(extract_radius_wrapper, seg_paths), total=len(seg_paths)))
    large_vessel_radius = np.median(radii)
    print(f"Large vessel radius: {large_vessel_radius}")

    # Pass 2: re-extract graphs and compute features with known median radius
    print("Extracting features...")
    seg_paths_with_radius = [(*p, large_vessel_radius) for p in seg_paths]
    with Pool(processes=2, maxtasksperchild=1) as pool:
        feature_list = list(tqdm(pool.imap(process_single_file, seg_paths_with_radius), total=len(seg_paths)))
    # feature_list = []
    # for ele in seg_paths_with_radius:
    #     feature_list.append(process_single_file(ele))
    df = pd.DataFrame(feature_list)
    df.to_csv(os.path.join(results_folder, "features.csv"), index=False)
    df.to_excel(os.path.join(results_folder, "features.xlsx"), index=False)
