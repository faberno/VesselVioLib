import os
from typing import Sequence, Optional
import numpy as np
import igraph as ig

from vvl.utils.image_processing import load_volume
from vvl.utils.io import save_graph
from vvl.analysis import extract_graph_from_volume, extract_graph_and_volume_features
from vvl.features import extract_radius


class GraphInfo:
    def __init__(
        self,
        vesselseg_path: str,
        layerseg_path: str,
        depth: int,
        resolution: Sequence[float],
        filter_length: int,
        prune_length: float,
        legacy:bool,
        output_dir: Optional[str] = None,
    ):
        self.volume_path = vesselseg_path
        self.name = os.path.basename(vesselseg_path).replace(".nii.gz", "")
        self.unfiltered_vol, _ = load_volume(vesselseg_path)
        self.legacy = legacy
        if legacy:
            self.unfiltered_vol = self.unfiltered_vol.swapaxes(0, 2)

        self.graph_features = {}
        self.large_vessel_radius = None

        self.resolution = resolution
        self.filter_length = filter_length
        self.prune_length = prune_length
        self.output_dir = output_dir

        self.filtered_vol = None
        self.nx_graph = None
        self.i_graph = None
        self.features = {"name": self.name}
        self.large_vessel_radius = None
        if layerseg_path:
            self.layerseg_vol, _ = load_volume(layerseg_path)
        else:
            self.layerseg_vol = None
        self.layer_depth_map = self.compute_layer_depth_map() if layerseg_path else None
        self.upper_lower_depth = depth

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

    def compute_layer_depth_map(self):
        lay = self.layerseg_vol
        depth_map = lay[..., ::-1].argmax(axis=2)
        if depth_map.min() == 0:
            print(self.name)
            print(
                "WARNING: Depth map contains zeros. THIS COULD MEAN THAT THERE ARE HOLES IN THE LAYER SEGMENTATION."
            )
        depth_map = lay.shape[2] - depth_map
        depth_map -= np.min(depth_map)
        self.depth_map = depth_map

    def vessel_depth_is_lower(self, x, y, z):
        x, y, z = int(round(x)) - 1, int(round(y)) - 1, int(round(z)) - 1
        z_offset = self.depth_map[x, y]
        z_offset += self.upper_lower_depth
        return z >= z_offset

    def prune_graph_upper_lower(self):
        # Create two new graphs of the same type as the original
        lower_graph = self.nx_graph.__class__()
        upper_graph = self.nx_graph.__class__()

        # Copy graph attributes
        lower_graph.graph.update(self.nx_graph.graph)
        upper_graph.graph.update(self.nx_graph.graph)

        # Sets to track which nodes should be included in each graph
        lower_nodes = set()
        upper_nodes = set()

        # Process edges and classify them
        for n1, n2, data in self.nx_graph.edges(data=True):
            original_edge_postions = data["original_edge_positions"]

            locs = [self.vessel_depth_is_lower(x, y, z) for x, y, z in original_edge_postions]
            is_lower = np.sum(locs) / len(locs)

            if is_lower:
                lower_graph.add_edge(n1, n2, **data)
                lower_nodes.add(n1)
                lower_nodes.add(n2)
            else:
                upper_graph.add_edge(n1, n2, **data)
                upper_nodes.add(n1)
                upper_nodes.add(n2)

        # Add node attributes for nodes that are in the graphs
        for node in lower_nodes:
            if node in self.nx_graph.nodes:
                lower_graph.add_node(node, **self.nx_graph.nodes[node])

        for node in upper_nodes:
            if node in self.nx_graph.nodes:
                upper_graph.add_node(node, **self.nx_graph.nodes[node])

        self.lower_graph = lower_graph
        self.upper_graph = upper_graph
        save_dir = os.path.join(self.output_dir, "Graphs")

        if self.output_dir is not None:
            save_graph(ig.Graph.from_networkx(self.lower_graph), self.name+"_lower", self.output_dir)
            save_graph(ig.Graph.from_networkx(self.upper_graph), self.name+"_upper", self.output_dir)


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
