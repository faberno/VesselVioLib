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


import logging
from math import sqrt
from pathlib import Path
from threading import Lock

import numpy as np
from numba import njit, prange

from vvl.utils import get_cwd, measure_time

logger = logging.getLogger(__name__)

######################
### LUT Generation ###
######################
@njit(parallel=True, cache=True)
def table_generation(resolution=np.array([1, 1, 1]), size=150):
    # size = min(500, ceil(max_radius / np.min(resolution))) # Hard code size limit at 500 mb
    LUT = np.zeros((size, size, size))

    correction = resolution / 2

    for z in prange(size):
        for y in range(size):
            for x in range(size):
                coords = np.array([z, y, x])
                coords = coords * resolution
                non_zeros = np.count_nonzero(coords)

                # To correct for radii lines along 1D planes, remove half of resolution length.
                if non_zeros == 1:
                    # Two of the values will be 0 and therefore negative after correction.
                    corrected = coords - correction
                    # Remove to isolate true correction.
                    corrected = corrected[corrected > 0][0]
                    LUT[z, y, x] = corrected

                else:
                    a = np.sum(coords**2)
                    LUT[z, y, x] = sqrt(a)
    return LUT

# Add a lock for file operations
file_lock = Lock()

###################
### LUT Loading ###
###################
# Load the corrections table.
def load_corrections(
    resolution=np.array([1., 1., 1.]),
    new_build=False,
    visualize=False,
    size=150,
):

    # Load the correct LUT: resolution(analysis) or basis(visualization) units.
    wd = Path(get_cwd())
    rc_path = wd / "library" / "volumes" / ("Vis_Radii_Corrections.npy" if visualize else "Radii_Corrections.npy")

    @measure_time
    def build_lut(resolution: np.ndarray):
        """Builds the LUT for the given resolution."""
        LUT = table_generation(resolution, size)

        # rc_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        # np.save(rc_path, LUT)

        return LUT

    if new_build or not rc_path.exists():
        logger.info("New build initiated.")
        LUT = build_lut(resolution)

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
