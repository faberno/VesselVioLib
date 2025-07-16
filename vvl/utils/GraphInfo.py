import os
from typing import Sequence, Optional
from vvl.utils.image_processing import load_volume

from vvl.utils.io import save_graph
from vvl.analysis import extract_graph_from_volume, extract_graph_and_volume_features
from vvl.features import extract_radius


class GraphInfo:
    def __init__(
        self,
        volume_path: str,
        resolution: Sequence[float],
        filter_length: int,
        prune_length: float,
        output_dir: Optional[str] = None,
    ):
        self.volume_path = volume_path
        self.name = os.path.basename(volume_path).replace(".nii.gz", "")
        self.unfiltered_vol, _ = load_volume(volume_path)
        self.resolution = resolution
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

    def vessel_depth_is_lower(self, x, y, z):
        pass

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
            coord_list = data["coords_list"]

            any_lower = any(vessel_depth_is_lower(x, y, z) for x, y, z in coord_list)
            any_upper = any(not vessel_depth_is_lower(x, y, z) for x, y, z in coord_list)

            if any_lower:
                lower_graph.add_edge(n1, n2, **data)
                lower_nodes.add(n1)
                lower_nodes.add(n2)
            if any_upper:
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
