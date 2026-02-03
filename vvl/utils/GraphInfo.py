import os
from typing import Sequence, Optional
import numpy as np
import igraph as ig
import pandas as pd
from vvl.utils.image_processing import load_volume
from vvl.utils.graph_processing import create_graph
from vvl.utils.io import save_graph
from vvl.analysis import extract_graph_from_volume, extract_graph_and_volume_features
from vvl.features import extract_radius, extract_int_features
from vvl.utils.volume_processing import (
    volume_prep,
    pad_volume,
    skeletonize,
    radii_calc_input,
)
from vvl.analysis import reconstruct_volume
import nibabel as nib


class GraphInfo:
    def __init__(
        self,
        vesselseg_path: str,
        layerseg_path: str,
        resolution: Sequence[float],
        filter_length: float,
        prune_length: float,
        legacy: bool,
        depth=None,
        output_dir: Optional[str] = None,
        structure_mask=None,
        normalize: bool = True,
    ):
        self.volume_path = vesselseg_path
        self.structure_mask = structure_mask
        self.recon = None
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
        self.normalize = normalize

        self.filtered_vol = None
        self.filtered_vol_lower = None
        self.filtered_vol_upper = None

        self.nx_graph = None
        self.i_graph = None
        self.features = {"name": self.name}
        self.large_vessel_radius = None
        if layerseg_path:
            self.layerseg_vol, _ = load_volume(layerseg_path)
            self.layerseg_vol = self.layerseg_vol.swapaxes(
                0, 2
            )  ##TODO shape verifizieren
            assert self.layerseg_vol.shape[0] < self.layerseg_vol.shape[2], (
                "Layer segmentation is wider than it is deep. Probably wrong axes used"
            )
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
        # TODO: In order to keep accurate TBV calculation, filtered vol should be depth cropped to final signal, this should also ignore floating vessels in background
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

    def flatten_coords(self, coords, surface, depth):
        coords = np.asarray(coords)
        yx = coords[:, [0, 1]].round().astype(int)
        shift = depth - surface[yx[:, 0], yx[:, 1]]
        coords[:, 2] += shift
        return coords

    def surfaceToMask(self, surface: np.ndarray, Nz: int, width: int = 1):
        Nx, Ny = surface.shape
        surface = np.clip(surface, 0, Nz - 1)
        mask = np.zeros(shape=(Nx, Ny, Nz), dtype=np.int8)
        for i in range(Nx):
            for j in range(Ny):
                mask[i, j, int(surface[i, j]) : int(surface[i, j] + width)] = 1
        return mask

    def compute_vp_depth(self):
        layseg = self.layerseg_vol.copy()
        layseg = np.array(layseg, dtype=np.float32)
        layseg = layseg.transpose(1, 0, 2)
        G = self.nx_graph
        nodes = []

        for n, d in G.nodes(data=True):
            x, y, z = (
                float(d["v_coords"][2]),
                float(d["v_coords"][1]),
                float(d["v_coords"][0]),
            )
            nodes.append([y, z, x])  # napari expects (z,y,x)

        junction_surface = (
            layseg.shape[2] - 1 - np.argmax(np.flip(layseg, axis=2), axis=2)
        )

        nodes_flat = self.flatten_coords(nodes, junction_surface, 0)
        z_vals = nodes_flat[:, 2]

        counts, bins = np.histogram(z_vals, bins=30)
        threshold = 0.3 * counts.max()
        z_thr = bins[np.where(counts >= threshold)[0][-1]]
        vp_depth = np.max(abs(z_vals)) - abs(z_thr)
        if vp_depth < 1:
            raise ValueError("VP depth cannot be zero")
        self.upper_lower_depth = int(vp_depth)

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

            locs = [
                self.vessel_depth_is_lower(x, y, z)
                for x, y, z in original_edge_postions
            ]
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

        if self.output_dir is not None:
            g = ig.Graph.from_networkx(self.lower_graph)
            tmp = [edge_positions for edge_positions in g.es["original_edge_positions"]]
            z_dists = [
                sum(sub_array[2] for sub_array in tmp_new) / len(tmp_new)
                for tmp_new in tmp
            ]
            x_dists = [
                sum(sub_array[0] for sub_array in tmp_new) / len(tmp_new)
                for tmp_new in tmp
            ]
            # x_dists = [self.depth_map.shape[0] - 1 - x for x in x_dists]
            y_dists = [
                sum(sub_array[1] for sub_array in tmp_new) / len(tmp_new)
                for tmp_new in tmp
            ]
            z_dists = [
                z_dists[i]
                - self.depth_map[int(round(x_dists[i])), int(round(y_dists[i]))]
                for i in range(len(z_dists))
            ]

            g.es["z_dist"] = z_dists
            save_graph(g, self.name + "_lower", self.output_dir)

            g = ig.Graph.from_networkx(self.upper_graph)
            try:
                tmp = [edge_positions for edge_positions in g.es["original_edge_positions"]]
                z_dists = [
                    sum(sub_array[2] for sub_array in tmp_new) / len(tmp_new)
                    for tmp_new in tmp
                ]
                x_dists = [
                    sum(sub_array[0] for sub_array in tmp_new) / len(tmp_new)
                    for tmp_new in tmp
                ]
                # x_dists = [self.depth_map.shape[0] - 1 - x for x in x_dists]
                y_dists = [
                    sum(sub_array[1] for sub_array in tmp_new) / len(tmp_new)
                    for tmp_new in tmp
                ]
                z_dists = [
                    z_dists[i]
                    - self.depth_map[int(round(x_dists[i])), int(round(y_dists[i]))]
                    for i in range(len(z_dists))
                ]

                g.es["z_dist"] = z_dists
            except:
                pass
            print(self.name + "_upper", self.output_dir)
            save_graph(g, self.name + "_upper", self.output_dir)

    def split_upper_lower_volume(self, save_vols=False):

        def save_nii(V, path):
            img = nib.Nifti1Image(V, np.eye(4))
            nib.save(img, path)
        volume, point_minima, point_maxima = volume_prep(self.filtered_vol)

        volume = pad_volume(volume)
        points = skeletonize(volume)

        # skeleton_radii, vis_radii = radii_calc_input(volume, points, self.resolution, gen_vis_radii=False, verbose=False)

        volume = volume[1:-1, 1:-1, 1:-1]
        points -= 1

        #         ig_graph_low = create_graph(
        #         volume.shape,
        #         skeleton_radii,
        #         vis_radii,
        #         points,
        #         point_minima,
        #         verbose=True
        # )

        filtered_lower = reconstruct_volume(volume, ig.Graph.from_networkx(self.lower_graph), points, self.resolution,
                                            point_minima, use_unreduced_nodes=True) >= 0
        filtered_upper = reconstruct_volume(volume, ig.Graph.from_networkx(self.upper_graph), points, self.resolution,
                                            point_minima, use_unreduced_nodes=True) >= 0
        self.filtered_vol_lower = filtered_lower
        self.filtered_vol_upper = filtered_upper

        if save_vols:
            save_nii(
                filtered_lower.astype(np.uint8),
                os.path.join(self.output_dir, self.name + "_lower.nii.gz"),
            )
            save_nii(
                filtered_upper.astype(np.uint8),
                os.path.join(self.output_dir, self.name + "_upper.nii.gz"),
            )

    def extract_features(self):
        assert len(self.features) == 1
        self.features.update(
            extract_graph_and_volume_features(
                G=self.nx_graph,
                volume=self.filtered_vol,
                resolution=self.resolution,
                large_vessel_radius=self.large_vessel_radius,
                structure_mask=self.structure_mask,
                normalization=self.normalize,
            )
        )

    def extract_features_upper_lower(self):
        print(self.name)
        assert len(self.features) == 1
        if self.upper_lower_depth is None:
            self.compute_vp_depth()
        self.features.update({"upper_lower_depth": self.upper_lower_depth})

        self.prune_graph_upper_lower()
        self.split_upper_lower_volume(save_vols=True)

        features_upper = extract_graph_and_volume_features(
            G=self.upper_graph.copy(),
            volume=self.filtered_vol_upper,
            resolution=self.resolution,
            large_vessel_radius=self.large_vessel_radius,
            structure_mask=self.structure_mask,
            normalization=self.normalize,
        )
        features_lower = extract_graph_and_volume_features(
            G=self.lower_graph.copy(),
            volume=self.filtered_vol_lower,
            resolution=self.resolution,
            large_vessel_radius=self.large_vessel_radius,
            structure_mask=self.structure_mask,
            normalization=self.normalize,
        )

        # features_int = extract_int_features(
        #     volume_lay=self.layerseg_vol,
        #     volume_upper=self.filtered_vol_upper,
        #     volume_lower=self.filtered_vol_lower,
        #     recon=self.recon,
        #     upper_lower_depth=self.upper_lower_depth,
        # )

        # append _lower and _upper suffix to corresponding dict keys
        features_upper = {
            key + "_upper": value for key, value in features_upper.items()
        }
        features_lower = {
            key + "_lower": value for key, value in features_lower.items()
        }
        self.features.update(features_upper)
        self.features.update(features_lower)
        # self.features.update(features_int)
