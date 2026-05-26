# single_file_processor.py
import numpy as np
import os
from pathlib import Path
import hdf5storage

from vvl.utils.GraphInfo import GraphInfo


# Configuration
vesselseg_path = r"e:\CVD_backup\deleteme\south_munich\DZMS\processed_aligned_corr_251203\vessel_segmentation\R_G015388800_ARM_Scan00001_img.nii.gz"  # Set this to your file path
layerseg_path = r"e:\CVD_backup\deleteme\south_munich\DZMS\processed_aligned_corr_251203\epidermis_segmentation\R_G015388800_ARM_Scan00001_img.nii.gz"

results_folder = os.path.join(os.path.dirname(vesselseg_path), "vesselvio")
Path(results_folder).mkdir(parents=True, exist_ok=True)

filter_length = 0.250  # remove paths with a length less than this
prune_length = 0.0  # remove connected endpoint vessels with length less than this
large_vessel_radius = 0.02  # Manually define at which radius vessels are considered large
vp_depth = 70  # Depth at which to seperate the vessels into upper and lower region
legacy_vessel = True


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




graph_info.extract_graph()

graph_info.large_vessel_radius = large_vessel_radius

graph_info.extract_features_upper_lower()

print("wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow wow")