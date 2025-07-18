import pytest
import numpy as np
from skimage.draw import line_nd, circle_perimeter
from scipy.ndimage import binary_dilation, distance_transform_edt
from pprint import pprint

from vvl.analysis import extract_graph_from_volume, extract_graph_and_volume_features

class TestGraphAndFeatureExtraction:
    def test_single_vertical_vessel_volume(self):

        N = (300, 200, 100)
        resolution = [0.003, 0.012, 0.012]
        start_point = (50, 100, 50)
        end_point = (250, 100, 50)
        radius = 0.025

        l = line_nd(start_point,
                    end_point,
                    endpoint=True)

        volume = np.zeros(N, dtype=bool)
        volume[l] = True

        edt = distance_transform_edt(~volume, sampling=resolution)

        volume = edt < radius

        _, graph, filtered_volume = extract_graph_from_volume(volume,
                                                              resolution=resolution,
                                                              filter_length=0.250,
                                                              prune_length=0)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

        nodes = [n[1] for n in graph.nodes(data=True)]
        edge = [e[2] for e in graph.edges(data=True)][0]

        print(f"Start Node -> Found: {nodes[0]['v_coords']} vs Expected: {start_point}")
        print(f"End Node -> Found: {nodes[1]['v_coords']} vs Expected: {end_point}")
        print(f"Length -> Found: {edge['length']} vs Expected: {np.linalg.norm((np.array(end_point) - np.array(start_point)) * resolution)}")
        print(f"Volume -> Found: {edge['volume']} vs Expected: {volume.sum() * np.prod(resolution)}")
        print(f"Tortuosity -> Found: {edge['tortuosity']} vs Expected: 1.0")
        print(f"Radius -> Found: {edge['radius_avg']} vs Expected: {radius}")

        features = extract_graph_and_volume_features(graph, filtered_volume, resolution=resolution, structure_mask=None,
                                                     normalization=False)
        pprint(features)

    def test_single_horizontal_vessel_volume(self):

        N = (300, 200, 100)
        resolution = [0.003, 0.012, 0.012]
        start_point = (150, 10, 90)
        end_point = (150, 190, 10)
        radius = 0.025

        l = line_nd(start_point,
                    end_point,
                    endpoint=True)

        volume = np.zeros(N, dtype=bool)
        volume[l] = True

        edt = distance_transform_edt(~volume, sampling=resolution)

        volume = edt < radius

        _, graph, filtered_volume = extract_graph_from_volume(volume,
                                                              resolution=resolution,
                                                              filter_length=0.250,
                                                              prune_length=0)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

        nodes = [n[1] for n in graph.nodes(data=True)]
        edge = [e[2] for e in graph.edges(data=True)][0]

        print(f"Start Node -> Found: {nodes[0]['v_coords']} vs Expected: {start_point}")
        print(f"End Node -> Found: {nodes[1]['v_coords']} vs Expected: {end_point}")
        print(f"Length -> Found: {edge['length']} vs Expected: {np.linalg.norm((np.array(end_point) - np.array(start_point)) * resolution)}")
        print(f"Volume -> Found: {edge['volume']} vs Expected: {volume.sum() * np.prod(resolution)}")
        print(f"Tortuosity -> Found: {edge['tortuosity']} vs Expected: 1.0")
        print(f"Radius -> Found: {edge['radius_avg']} vs Expected: {radius}")

        features = extract_graph_and_volume_features(graph, filtered_volume, resolution=resolution, structure_mask=None,
                                                     normalization=False)
        pprint(features)

    def test_single_vessel_with_noise_volume(self):

        N = (300, 200, 100)
        resolution = [0.003, 0.012, 0.012]
        start_point = (150, 10, 90)
        end_point = (150, 190, 10)
        radius = 0.025

        l = line_nd(start_point,
                    end_point,
                    endpoint=True)

        noise_line_1 = line_nd((100, 20, 30), (105, 22, 35), endpoint=True)
        noise_line_2 = line_nd((200, 180, 70), (210, 182, 70), endpoint=True)

        volume = np.zeros(N, dtype=bool)
        volume[l] = True
        volume[noise_line_1] = True
        volume[noise_line_2] = True

        edt = distance_transform_edt(~volume, sampling=resolution)

        volume = edt < radius

        _, graph, filtered_volume = extract_graph_from_volume(volume,
                                                              resolution=resolution,
                                                              filter_length=0.250,
                                                              prune_length=0)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

        nodes = [n[1] for n in graph.nodes(data=True)]
        edge = [e[2] for e in graph.edges(data=True)][0]

        print(f"Start Node -> Found: {nodes[0]['v_coords']} vs Expected: {start_point}")
        print(f"End Node -> Found: {nodes[1]['v_coords']} vs Expected: {end_point}")
        print(f"Length -> Found: {edge['length']} vs Expected: {np.linalg.norm((np.array(end_point) - np.array(start_point)) * resolution)}")
        print(f"Volume -> Found: {edge['volume']} vs Expected: {volume.sum() * np.prod(resolution)}")
        print(f"Tortuosity -> Found: {edge['tortuosity']} vs Expected: 1.0")
        print(f"Radius -> Found: {edge['radius_avg']} vs Expected: {radius}")

        features = extract_graph_and_volume_features(graph, filtered_volume, resolution=resolution, structure_mask=None,
                                                     normalization=False)
        pprint(features)

    def test_complex_vessel_volume(self):

        N = (300, 200, 100)
        resolution = [0.003, 0.012, 0.012]
        large_radius = 0.035
        small_radius = 0.015

        circle = circle_perimeter(150, 100, 30, shape=N)
        circle = (*circle, np.ones_like(circle[0], dtype=int) * 50)
        circle_volume = np.zeros(N, dtype=bool)
        circle_volume[circle] = True
        edt = distance_transform_edt(~circle_volume, sampling=resolution)
        circle_volume = edt < large_radius

        # add 2 vessels to circle
        line_volume = np.zeros(N, dtype=bool)
        line_volume[50:120, 100, 50] = True
        line_volume[180:250, 100, 50] = True
        edt = distance_transform_edt(~line_volume, sampling=resolution)
        line_volume = edt < small_radius

        volume = circle_volume | line_volume

        _, graph, filtered_volume = extract_graph_from_volume(volume,
                                                              resolution=resolution,
                                                              filter_length=0.250,
                                                              prune_length=0)
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 4

        nodes = [n[1] for n in graph.nodes(data=True)]
        edge = [e[2] for e in graph.edges(data=True)]

        features = extract_graph_and_volume_features(graph, filtered_volume, resolution=resolution, structure_mask=None,
                                                     normalization=False)

        # zero cycles will be found as it only consists of nodes and parallel edges are not allowed in cycle calculation
        pprint(features)