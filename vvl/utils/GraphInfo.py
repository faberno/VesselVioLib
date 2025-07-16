import os
from typing import Sequence, Optional
import nibabel as nib
import numpy as np

from vvl.utils.io import save_graph
from vvl.analysis import extract_graph_from_volume
from vvl.features import extract_features, extract_sizedependent_features


class GraphInfo:
    def __init__(
        self,
        volume_path: str,
        resolution: Sequence[float],
        vol_spacing: np.array,
        filter_length: int,
        prune_length: float,
        output_dir: Optional[str] = None,
    ):
        self.volume_path = volume_path
        self.name = os.path.basename(volume_path).replace(".nii.gz","")
        self.unfiltered_vol = nib.load(volume_path).get_fdata()
        self.resolution = resolution
        self.vol_spacing = vol_spacing
        self.filter_length = filter_length
        self.prune_length = prune_length
        self.output_dir = output_dir
        self.filtered_vol = None
        self.nx_graph = None
        self.i_graph = None
        self.features = {"name":self.name}
        self.size_features = None
        self.large_vessel_radius = None

    def extract_graph(self):
        graph, filtered_vol = extract_graph_from_volume(
            self.volume_path, self.resolution, self.filter_length, self.prune_length
        )

        if self.output_dir is not None:
            save_graph(graph, self.name, self.output_dir)

        self.i_graph = graph
        self.nx_graph = graph.to_networkx()
        self.filtered_vol = filtered_vol

    def extract_features(self):
        feats = extract_features(
            self.nx_graph,
            self.unfiltered_vol,
            self.filtered_vol,
            vol_spacing=self.vol_spacing,
        )
        self.features.update(feats)

    def extract_size_features(self):
        size_feats = extract_sizedependent_features(self.nx_graph, large_vessel_radius=self.large_vessel_radius)
        self.features.update(size_feats)