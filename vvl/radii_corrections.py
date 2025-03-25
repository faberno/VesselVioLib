"""
The Euclidean Distance Transformation (EDT) algorithm equation calculates point distance between the centers of voxels/pixels.
Because of this, radius of vessel segments is overestimated, as the true 'edge' of the vessel exists at the border of the nearest black voxel, not the center.

This program calculates the EDT to the edge of face-connected voxels rather than their centers. Edge and corner voxel EDTS are computed normally.

The program then generates an array of a predetermined size that can be accessed to rapidly provide corrected distances for our centerlines.

Copyright 2022, Jacob Bumgarner
"""

__author__ = "Jacob Bumgarner <jrbumgarner@mix.wvu.edu>"
__license__ = "GPLv3 - GNU General Pulic License v3 (see LICENSE)"
__copyright__ = "Copyright 2022 by Jacob Bumgarner"
__webpage__ = "https://jacobbumgarner.github.io/VesselVio/"
__download__ = "https://jacobbumgarner.github.io/VesselVio/Downloads"


import os
from math import sqrt
from threading import Lock
from time import perf_counter as pf
from pathlib import Path

import numpy as np
from numba import njit, prange

from vvl import logger


######################
### LUT Generation ###
######################

# ORIGINAL IMPLEMENTATION

# @njit(parallel=True, cache=True)
# def table_generation(resolution=np.array([1, 1, 1]), size=150):
#     # size = min(500, ceil(max_radius / np.min(resolution))) # Hard code size limit at 500 mb
#     LUT = np.zeros((size, size, size))
#
#     correction = resolution / 2
#
#     for z in prange(size):
#         for y in range(size):
#             for x in range(size):
#                 coords = np.array([z, y, x])
#                 coords = coords * resolution
#                 non_zeros = np.count_nonzero(coords)
#
#                 # To correct for radii lines along 1D planes, remove half of resolution length.
#                 if non_zeros == 1:
#                     # Two of the values will be 0 and therefore negative after correction.
#                     corrected = coords - correction
#                     # Remove to isolate true correction.
#                     corrected = corrected[corrected > 0][0]
#                     LUT[z, y, x] = corrected
#
#                 else:
#                     a = np.sum(coords**2)
#                     LUT[z, y, x] = sqrt(a)
#     return LUT


def table_generation(resolution=np.array([1, 1, 1]), size=150):
    coord1d = np.arange(size, dtype=np.float64)
    coords = np.stack(
        np.meshgrid(coord1d, coord1d, coord1d, indexing="ij"), axis=-1
    ) * resolution
    LUT = np.linalg.norm(coords, axis=-1)

    # To correct for radii lines along 1D planes, remove half of resolution length.
    voxel_1dplane = np.count_nonzero(coords, axis=-1) == 1
    correction = resolution / 2
    corrected = coords[voxel_1dplane] - correction

    # Two of the values will be 0 and therefore negative after correction.
    LUT[voxel_1dplane] = corrected.max(1)
    return LUT

# Add a lock for file operations
file_lock = Lock()

###################
### LUT Loading ###
###################
# Load the corrections table.
def load_corrections(
    resolution=np.array([1, 1, 1]),
    new_build=False,
    visualize=False,
    size=150,
):

    # Load the correct LUT: resolution(analysis) or basis(visualization) units.
    wd = Path(get_cwd())  # Find wd
    if not visualize:
        rc_path = wd / "library" / "volumes" / "Radii_Corrections.npy"
    else:
        rc_path = wd / "library" / "volumes" / "Vis_Radii_Corrections.npy"

    # Build function
    def build(resolution):
        start_time = pf()

        logger.info("Generating new correction table.")
        LUT = table_generation(resolution, size)

        rc_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rc_path, LUT)

        logger.info(f"Table generated in {(pf() - start_time):.2f} s.")
        return LUT

    if new_build or not rc_path.exists():
        logger.info("New build initiated.")
        LUT = build(resolution)

    else:
        try:
            with file_lock:  # Use lock to prevent concurrent access
                # Try loading the file.
                LUT = np.load(rc_path)

            rebuild = False

            # Make sure dimensions are also correct.
            if (
                resolution[0] / 2 != LUT[1, 0, 0]
                or resolution[1] / 2 != LUT[0, 1, 0]
                or resolution[2] / 2 != LUT[0, 0, 1]
            ):
                rebuild = True
            if rebuild:
                LUT = build(resolution)

        except Exception as error:
            print(f"Unexpected error encountered: {error}")
            LUT = build(resolution)

    return LUT


######################
### Terminal Build ###
######################
if __name__ == "__main__":
    from os import getcwd as get_cwd  # Can't load helpers from Library level

    resolution = 1.0  # Either a float or an array.
    load_corrections(resolution, new_build=True, verbose=True, Visualize=False)
    load_corrections(resolution, new_build=True, verbose=True, Visualize=True)
else:
    from vvl.helpers import get_cwd
