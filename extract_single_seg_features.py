# single_file_processor.py
import numpy as np
import os
from pathlib import Path
import hdf5storage

from vvl.utils.GraphInfo import GraphInfo


# Configuration
vesselseg_path = r"e:\CVD_backup\south_munich\DZMS\processed\vessel_preds\pred003\R_G058957050_ARM_Scan00001_img_corr_.nii.gz"  # Set this to your file path
layerseg_path = r"e:\CVD_backup\south_munich\DZMS\processed\lay_pred\select\R_G058957050_ARM_Scan00001_img_corr__processed.nii.gz"

lf_p = r"e:\CVD_backup\south_munich\DZMS\processed\recon\R_G058957050_ARM_Scan00001_img_corr_LF.mat"
hf_p = r"e:\CVD_backup\south_munich\DZMS\processed\recon\R_G058957050_ARM_Scan00001_img_corr_HF.mat"

results_folder = os.path.join(os.path.dirname(vesselseg_path), "vesselvio")
Path(results_folder).mkdir(parents=True, exist_ok=True)

filter_length = 0.250  # remove paths with a length less than this
prune_length = 0.0  # remove connected endpoint vessels with length less than this
large_vessel_radius = 14.4  # Manually define at which radius vessels are considered large
vp_depth = 70  # Depth at which to seperate the vessels into upper and lower region
legacy_vessel = False # 


resolution = [0.012, 0.012, 0.003] 

graph_info = GraphInfo(
    vesselseg_path,
    layerseg_path,
    resolution=resolution,
    filter_length=filter_length,
    prune_length=prune_length,
    legacy=legacy_vessel,
    output_dir=results_folder,
)

matfile_hf = hdf5storage.loadmat(
    hf_p, options=hdf5storage.Options(structs_as_dicts=True)
)
hf = np.array(matfile_hf["R"]).swapaxes(0, 1)

matfile_lf = hdf5storage.loadmat(
    lf_p, options=hdf5storage.Options(structs_as_dicts=True)
)
lf = np.array(matfile_lf["R"]).swapaxes(0, 1)

graph_info.recon = np.stack([lf, hf], axis=0)


graph_info.extract_graph()

graph_info.large_vessel_radius = large_vessel_radius

graph_info.extract_features_upper_lower()

print("wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow")