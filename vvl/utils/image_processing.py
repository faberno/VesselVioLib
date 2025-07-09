import os
from time import perf_counter as pf

import numpy as np
import nibabel
from skimage.io import imread

from vvl.utils.helpers import get_ext


def get_filename(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return filename

def prep_resolution(resolution):
    if not isinstance(resolution, list):
        resolution = np.repeat(resolution, 3)
    else:
        # Flip the resolution, as numpy first index will represent image depth
        resolution = np.array(resolution)
    min_resolution = np.min(resolution)
    return resolution


def load_volume(file, verbose=False):
    t1 = pf()

    # Only use .nii files for annotations, this is mainly due to loading speeds
    if get_ext(file) == ".nii" or get_ext(file) == ".gz":
        try:
            volume = load_nii_volume(file)
        except Exception as error:
            print(f"Could not load .nii file using nibabel: {error}")
            volume = skimage_load(file)
    else:
        volume = skimage_load(file)

    if volume is None or volume.ndim not in (2, 3):
        return None

    if verbose:
        print(f"Volume loaded in {pf() - t1:.2f} s.")

    return volume, volume.shape

# Load nifti files
def load_nii_volume(file):
    proxy = nibabel.load(file)
    data = proxy.dataobj.get_unscaled().transpose()
    if data.ndim == 4:
        data = data[0]
    return data


# Load an image volume using SITK, return None upon read failure
def skimage_load(file):
    try:
        volume = imread(file).astype(np.uint8)
    except Exception as error:
        print(f"Unable to read image file using skimage.io.imread: {error}")
        volume = None
    return volume

def binary_check(volume: np.ndarray) -> bool:
    """Return a bool indicating if the loaded volume is binary or not.

    Takes a slice from the volume and checks to confirm that only two unique
    values are present.

    Parameters:
    volume : np.ndarray

    Returns:
    bool
        True if the spot check of the volume only return two unique values,
        False if more than two unique values were identified.
    """
    middle = int(volume.shape[0] / 2)
    unique = np.unique(volume[middle])

    return unique.shape[0] < 3

# Reshape 2D array to make it compatible with analysis pipeline
def reshape_2D(points, volume, verbose=False):
    if verbose:
        print("Re-constructing arrays...", end="\r")
    points = np.pad(points, ((0, 0), (1, 0)))
    zeros = np.zeros(volume.shape)  # Pad zeros onto back of array
    volume = np.stack([volume, zeros])
    image_shape = volume.shape
    return points, volume, image_shape