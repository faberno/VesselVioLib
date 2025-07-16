import os
from typing import Sequence, Optional
import nibabel as nib
import numpy as np
import networkx as nx

from vvl.utils.io import save_graph
from vvl.analysis import extract_graph_from_volume, extract_graph_and_volume_features
from vvl.features import extract_radius


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
        self.name = os.path.basename(volume_path).replace(".nii.gz", "")
        self.unfiltered_vol = nib.load(volume_path).get_fdata()
        self.resolution = resolution
        self.vol_spacing = vol_spacing
        self.filter_length = filter_length
        self.prune_length = prune_length
        self.output_dir = output_dir

        self.filtered_vol = None
        self.nx_graph = None
        self.i_graph = None
        self.features = {"name": self.name}
        self.large_vessel_radius = None

    def extract_graph(self):
        graph, graph_nx, filtered_vol = extract_graph_from_volume(
            self.unfiltered_vol, self.resolution, self.filter_length, self.prune_length
        )

        if self.output_dir is not None:
            save_graph(graph, self.name, self.output_dir)

        self.nx_graph = graph_nx
        self.i_graph = graph
        self.filtered_vol = filtered_vol

    def extract_radius(self):
        if self.nx_graph is None:
            try:
                self.extract_graph()
            except Exception as e:
                print(f"Error extracting graph: {e}")
                raise

        return extract_radius(self.nx_graph)

    def extract_features(self):
        self.features.update(
            extract_graph_and_volume_features(
                G=self.nx_graph,
                volume=self.filtered_vol,
                resolution=self.resolution,
                large_vessel_radius=self.large_vessel_radius,
                structure_mask=None,
            )
        )
