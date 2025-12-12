import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd
from util.util_jk import loadmat
from vvl.utils.GraphInfo import GraphInfo

vesselseg_dir = r"C:\Users\go34nuy\data\Repro\Positions\processed\vessel_segmentation"
layerseg_dir = r"C:\Users\go34nuy\data\Repro\Positions\processed\epidermis_segmentation"

filter_length = 0.250  # remove paths with a length less than this
prune_length = 0.0  # remove connected endpoint vessels with length less than this
large_vessel_radius = 0.018  # Manually define at which radius vessels are considered large
vp_depth = 40  # Depth at which to seperate the vessels into upper and lower region
legacy = True  # If using new vesselseg like synthetic vesselseg this Flag needs to be set to true
normalize = False # If vessel signal is already cropped to normalized volume then don't need to normalize
#Large vessel radius is 0.015997390253347205

resolution = [0.012, 0.012, 0.003] 



def find_layseg_for_vesseg(vesseg_path, layseg_dir):
    def clean_up_name(name):
        name = name.replace("_processed", "")
        name = name.replace("__l", "")
        name = name.replace("_l", "")
        name = name.replace("_0000", "")
        name = name.replace("_0001", "")
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

def save_graph_wrapper(g_i):
    g_i.save_graphs()

def extract_radii_wrapper(g_i):
    return g_i.extract_radius()

def extract_feats_wrapper(g_i):
    g_i.extract_features()
    return g_i

def extract_sizefeats_wrapper(g_i):
    # g_i.extract_features()
    g_i.extract_features_upper_lower()

    return g_i


if __name__ == "__main__":
    results_folder = os.path.join(vesselseg_dir, "vesselvio")
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    graph_infos = []
    # Load all files
    print("Loading files...")
    segs = [seg for seg in os.listdir(vesselseg_dir) if ".nii.gz" in seg]
    for vesselseg_name in tqdm(segs):
        vesselseg_path = os.path.join(vesselseg_dir, vesselseg_name)

        p = Path(vesselseg_path)
        base = p.stem[:-4] 
        recon_lf_path = p.parents[1] / "recon" / f"{base}LF.mat"
        recon_hf_path = p.parents[1] / "recon" / f"{base}HF.mat"
        recon_lf = loadmat(recon_lf_path)['R']
        recon_hf = loadmat(recon_hf_path)['R']

        graph_info = GraphInfo(
            vesselseg_path,
            find_layseg_for_vesseg(vesselseg_path, layerseg_dir),
            recon = [recon_lf, recon_hf],
            resolution=resolution,
            filter_length=filter_length,
            prune_length=prune_length,
            legacy=legacy,
            output_dir=results_folder,
            normalize=normalize,
        )
        graph_infos.append(graph_info)

    # Extract Graphs
    print("Extracting graphs...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(extract_graph_wrapper, graph_infos), total=len(graph_infos)))
    graph_infos.clear()
    graph_infos.extend(results)

    # for g_i in tqdm(graph_infos):
    #     g_i.extract_graph()

    # Extract radii
    print("Extracting radii...")
    with Pool() as pool:
        radii = list(tqdm(pool.imap(extract_radii_wrapper, graph_infos), total=len(graph_infos)))

    large_vessel_radius = np.median(radii)

    print(f"Large vessel radius is {large_vessel_radius}")
    for g_i in graph_infos:
        g_i.large_vessel_radius = large_vessel_radius

    # Extract size-dependent features
    print("Extracting size-dependent  features...")
    # with Pool() as pool:
    #     results = list(
    #         tqdm(pool.imap(extract_sizefeats_wrapper, graph_infos), total=len(graph_infos))
    #     )
    # graph_infos.clear()
    # graph_infos.extend(results)

    feature_list = []
    log_path = os.path.join(results_folder, "log.txt")
    for g_i in tqdm(graph_infos):
        try:  
            g_i.extract_features_upper_lower()
            feature_list.append(g_i.features)
            # df = pd.DataFrame(g_i.features, index=[0])
            # df.to_csv(os.path.join(results_folder, f"{g_i.name} features.csv"), index=False)
        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"{g_i.name} | {type(e).__name__}: {e}\n")
                print("FAILED: ", e)

    # feature_list = [g_i.features for g_i in graph_infos]


    df = pd.DataFrame(feature_list)
    df.to_csv(os.path.join(results_folder, "features.csv"), index=False)
    df.to_excel(os.path.join(results_folder, "features.xlsx"), index=False)



    print("Saving graphs...")
    with Pool(4) as pool:
        try: 
            _ = list(tqdm(pool.imap(save_graph_wrapper, graph_infos), total=len(graph_infos)))
        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"{g_i.name} | {type(e).__name__}: {e}\n")
                print("FAILED: ", e)
