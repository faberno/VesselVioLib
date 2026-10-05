import pickle

import igraph as ig
import nibabel as nib
import numpy as np
import pytest

from vvl.utils.reconstruct_graph_volume_erik import (
    ReconstructionConfig,
    inferred_output_resolution,
    load_vesselvio_pickle,
    reconstruct_from_pickle,
    reconstruct_pickle_to_nifti,
)
from vvl.utils.volume_processing import pad_volume, skeletonize, volume_prep


def _make_source_and_saved_graph():
    source = np.zeros((12, 12, 12), dtype=np.uint8)
    source[2:10, 4:7, 4:7] = 1

    bounded, minima, _maxima = volume_prep(source.copy())
    points = skeletonize(pad_volume(bounded))
    saved_positions = points + np.asarray(minima) - 1

    graph = ig.Graph(n=2, edges=[(0, 1)])
    graph.vs["v_coords"] = [saved_positions[0], saved_positions[-1]]
    graph.es["original_edge_positions"] = [saved_positions]
    return source, graph


def test_reconstruction_uses_vvl_original_edge_positions():
    source, graph = _make_source_and_saved_graph()
    config = ReconstructionConfig(resolution=(1.0, 1.0, 1.0))

    reconstructed, stats = reconstruct_from_pickle(
        graph,
        source,
        config=config,
    )

    np.testing.assert_array_equal(reconstructed, source)
    assert stats.input_foreground_voxels == np.count_nonzero(source)
    assert stats.output_foreground_voxels == np.count_nonzero(source)
    assert stats.retained_fraction == pytest.approx(1.0)


def test_legacy_axis_swap_is_reversed_for_output():
    source, _unused_graph = _make_source_and_saved_graph()
    working_source = source.swapaxes(0, 2)
    bounded, minima, _maxima = volume_prep(working_source.copy())
    points = skeletonize(pad_volume(bounded))
    saved_positions = points + np.asarray(minima) - 1
    graph = ig.Graph(n=2, edges=[(0, 1)])
    graph.vs["v_coords"] = [saved_positions[0], saved_positions[-1]]
    graph.es["original_edge_positions"] = [saved_positions]

    reconstructed, _ = reconstruct_from_pickle(
        graph,
        source,
        config=ReconstructionConfig(
            resolution=(1.0, 1.0, 1.0),
            legacy_swap_axes=True,
        ),
    )

    np.testing.assert_array_equal(reconstructed, source)


def test_missing_original_edge_positions_is_rejected(tmp_path):
    graph = ig.Graph(n=2, edges=[(0, 1)])
    pickle_path = tmp_path / "incomplete.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(graph, file)

    with pytest.raises(ValueError, match="original_edge_positions"):
        load_vesselvio_pickle(pickle_path)


def test_output_resolution_accounts_for_legacy_axis_swap():
    normal = ReconstructionConfig(
        resolution=(0.012, 0.006, 0.003),
        legacy_swap_axes=False,
    )
    legacy = ReconstructionConfig(
        resolution=(0.012, 0.006, 0.003),
        legacy_swap_axes=True,
    )

    assert inferred_output_resolution(normal) == (0.012, 0.006, 0.003)
    assert inferred_output_resolution(legacy) == (0.003, 0.006, 0.012)


def test_complete_workflow_ignores_source_nifti_metadata(tmp_path):
    source, graph = _make_source_and_saved_graph()
    pickle_path = tmp_path / "graph.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(graph, file)

    source_affine = np.array(
        [
            [0.0, -7.0, 0.0, 100.0],
            [8.0, 0.0, 0.0, -50.0],
            [0.0, 0.0, 9.0, 25.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    segmentation_path = tmp_path / "segmentation.nii.gz"
    nib.save(nib.Nifti1Image(source, source_affine), segmentation_path)
    output_path = tmp_path / "reconstructed.nii.gz"
    config = ReconstructionConfig(resolution=(0.4, 0.5, 0.6))

    reconstruct_pickle_to_nifti(
        pickle_path,
        segmentation_path,
        output_path,
        config=config,
    )

    result = nib.load(output_path)
    expected_affine = np.diag([0.4, 0.5, 0.6, 1.0])
    np.testing.assert_allclose(result.affine, expected_affine)
    assert result.shape == source.shape
    assert result.get_data_dtype() == np.dtype(np.uint8)
    np.testing.assert_array_equal(np.asanyarray(result.dataobj), source)
