# single_file_processor.py
import numpy as np
import os
from pathlib import Path

from vvl.utils.GraphInfo import GraphInfo


# Configuration
vesselseg_path = r"E:\CVD_backup\south_munich\DZMS\processed\uncorrected\vessel_preds\preds005\R_G058957050_ARM_Scan00001_img_.nii.gz"  # Set this to your file path
layerseg_path = r"E:\CVD_backup\south_munich\DZMS\processed\uncorrected\layer_segmentation\processed\good_qual\R_G058957050_ARM_Scan00001_img__processed.nii.gz"

results_folder = os.path.join(os.path.dirname(vesselseg_path), "vesselvio")
Path(results_folder).mkdir(parents=True, exist_ok=True)

# resolution = [0.003, 0.012, 0.012] # Fabian axes order x,y,z)
resolution = [0.012, 0.012, 0.003]  # Legacy axes order (z,y,x)
filter_length = 0.250
prune_length = 0.0
large_vessel_radius = 14.4


graph_info = GraphInfo(
    vesselseg_path,
    layerseg_path,
    depth=70,
    resolution=resolution,
    filter_length=filter_length,
    prune_length=prune_length,
    output_dir=results_folder,
)

graph_info.extract_graph()

graph_info.large_vessel_radius = large_vessel_radius

graph_info.prune_graph_upper_lower()

graph_info.extract_features()
