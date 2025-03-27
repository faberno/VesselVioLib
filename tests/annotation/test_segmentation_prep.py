import os
import sys

import pytest

sys.path.insert(1, "/Users/jacobbumgarner/Documents/GitHub/VesselVio")

import numba
import numpy as np

from vvl.annotation import segmentation_prep, tree_processing

THIS_PATH = os.path.realpath(__file__)
FIXTURE_DIR = os.path.join(os.path.dirname(THIS_PATH), "test_files")
ANNOTATION_DIR = os.path.join(FIXTURE_DIR, "annotation_data")


@pytest.mark.datafiles(ANNOTATION_DIR)
def test_build_roi_array(datafiles):
    annotation_dict = tree_processing.load_annotation_file(
        os.path.join(datafiles, "Cortex Unique.json")
    )

    # ID Check
    roi_array = segmentation_prep.build_roi_array(annotation_dict, annotation_type="ID")
    assert roi_array.shape == (6, 78)
    assert np.issubdtype(roi_array.dtype, np.uint32)

    # RGB Check
    roi_array = segmentation_prep.build_roi_array(
        annotation_dict, annotation_type="RGB"
    )
    assert roi_array.shape == (6, 2)
    assert np.issubdtype(roi_array.dtype, np.uint32)
    print(roi_array)
    return


@pytest.mark.datafiles(ANNOTATION_DIR)
def test_convert_hex_list_to_int(datafiles):
    annotation_dict = tree_processing.load_annotation_file(
        os.path.join(datafiles, "Cortex Unique.json")
    )
    hex_ROIs = [annotation_dict[key]["colors"] for key in annotation_dict.keys()]
    int_ROIs = segmentation_prep.convert_hex_list_to_int(hex_ROIs)
    assert len(int_ROIs) == 6
    assert len(int_ROIs[0]) == 1
    assert isinstance(int_ROIs[0][0], int)


test_children = [("Max Children Test.json", 41), ("Cortex Unique.json", 1)]


@pytest.mark.parametrize("datafile, expected_length", test_children)
def test_find_max_children_count(datafile, expected_length):
    annotation_data = tree_processing.load_annotation_file(
        os.path.join(ANNOTATION_DIR, datafile)
    )

    ROIs = [annotation_data[key]["colors"] for key in annotation_data.keys()]
    max_len = segmentation_prep.find_max_children_count(ROIs)
    assert max_len == expected_length

    return


test_roi_prep = [("Max Children Test.json", 552), ("Cortex Unique.json", 151)]


@pytest.mark.parametrize("datafile, expected_keys", test_roi_prep)
def test_prep_roi_array(datafile, expected_keys):
    annotation_data = tree_processing.load_annotation_file(
        os.path.join(ANNOTATION_DIR, datafile)
    )

    roi_array = segmentation_prep.build_roi_array(annotation_data, "ID")
    id_dict, id_keys = segmentation_prep.prep_roi_array(roi_array)
    assert isinstance(id_dict, numba.typed.typeddict.Dict)
    assert isinstance(id_keys, set)
    assert len(id_keys) == expected_keys
    return


@pytest.mark.datafiles(ANNOTATION_DIR)
def test_prep_volume_arrays(datafiles):
    annotation_data = tree_processing.load_annotation_file(
        os.path.join(datafiles, "Cortex Unique.json")
    )

    roi_array = segmentation_prep.build_roi_array(annotation_data, "ID")
    roi_volumes, volume_updates = segmentation_prep.prep_volume_arrays(roi_array)

    assert isinstance(roi_volumes, np.ndarray)
    assert roi_volumes.shape == (6,)
    assert volume_updates.shape == (6, 6)


def test_build_minima_maxima_arrays():
    volume = np.zeros((5, 10, 10))
    roi_array = np.zeros((10, 5))

    minima, maxima = segmentation_prep.build_minima_maxima_arrays(volume, roi_array)
    assert minima.shape == (roi_array.shape[0], 3)
    assert maxima.shape == (roi_array.shape[0], 3)

    assert np.all(minima[0] == volume.shape)
    assert np.all(maxima[0] == (0, 0, 0))

    return
