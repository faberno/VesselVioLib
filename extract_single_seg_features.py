# single_file_processor.py
import numpy as np
from vvl.utils.GraphInfo import GraphInfo


# Configuration
volume_path = r"D:\data\joshua\Atlas\vessel_preds\vp\preds038\R_20241106145816_ATLAS001_532nm_uperarm1_RSOM50_corr.nii.gz"  # Set this to your file path

resolution = [0.003, 0.012, 0.012]
filter_length = 0.250
prune_length = 0.0
large_vessel_radius = 14.4


graph_info = GraphInfo(
    volume_path,
    resolution=resolution,
    filter_length=filter_length,
    prune_length=prune_length,
)

graph_info.extract_graph()

graph_info.large_vessel_radius = large_vessel_radius

graph_info.extract_features()

