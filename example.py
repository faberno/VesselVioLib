import logging
logging.basicConfig(level=logging.INFO)

from vvl.analysis import extract_graph_from_volume, extract_graph_and_volume_features
from vvl.utils.image_processing import load_volume
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\fabia\PycharmProjects\VesselVioLib\data\2218_mask.nii.gz"

volume, _ = load_volume(path)

structure_mask = np.zeros(volume.shape, dtype=bool)
structure_mask[500:] = True

volume[~structure_mask] = 0

graph, filtered_volume = extract_graph_from_volume(volume,
                                                   resolution=[0.003, 0.012, 0.012],
                                                   filter_length=0.250,
                                                   prune_length=0)

features = extract_graph_and_volume_features(graph, filtered_volume, resolution=[0.003, 0.012, 0.012], structure_mask=structure_mask)
