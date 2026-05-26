import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from vvl.utils.GraphInfo import GraphInfo

vesselseg_dir = r"E:\CVD_backup\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\vessel\arm"
layerseg_dir = r"E:\CVD_backup\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\epidermis"

filter_length = 0.250  # remove paths with a length less than this
prune_length = 0.0  # remove connected endpoint vessels with length less than this
large_vessel_radius = 0.020  # Static dataset-wide median (e.g. TARTU baselines)
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

    # Whole-volume features (no upper/lower split)
    g_i.extract_features()
    features_whole = dict(g_i.features)

    # Reset features dict so extract_features_upper_lower's assert passes
    g_i.features = {"name": g_i.name}
    g_i.extract_features_upper_lower()
    features_upper_lower = dict(g_i.features)

    return features_whole, features_upper_lower


if __name__ == "__main__":
    results_folder = os.path.join(vesselseg_dir, "vesselvio")
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    segs = [seg for seg in os.listdir(vesselseg_dir) if ".nii.gz" in seg]
    seg_paths = [
        (
            os.path.join(vesselseg_dir, seg),
            find_layseg_for_vesseg(os.path.join(vesselseg_dir, seg), layerseg_dir),
            results_folder,
            large_vessel_radius,
        )
        for seg in segs
    ]

    print(f"Extracting features with static large_vessel_radius={large_vessel_radius}...")
    with Pool(processes=6, maxtasksperchild=1) as pool:
        results = list(tqdm(pool.imap(process_single_file, seg_paths), total=len(seg_paths)))

    whole_list = [r[0] for r in results]
    upper_lower_list = [r[1] for r in results]

    df_whole = pd.DataFrame(whole_list)
    df_whole.to_csv(os.path.join(results_folder, "features_whole.csv"), index=False)
    df_whole.to_excel(os.path.join(results_folder, "features_whole.xlsx"), index=False)

    df_ul = pd.DataFrame(upper_lower_list)
    df_ul.to_csv(os.path.join(results_folder, "features_upperLower.csv"), index=False)
    df_ul.to_excel(os.path.join(results_folder, "features_upperLower.xlsx"), index=False)
