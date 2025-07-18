import os
from math import sqrt
from threading import Lock
from pathlib import Path

import numpy as np
from numba import njit, prange

from vvl.utils.helpers import get_cwd


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
    resolution=np.array([1, 1, 1]),
    new_build=False,
    Visualize=False,
    size=150,
    verbose=False,
):

    # Load the correct LUT: resolution(analysis) or basis(visualization) units.
    wd = Path(os.path.abspath(__file__)).parent
    if not Visualize:
        rc_path = Path(os.path.abspath(__file__)).parents[2] / "library" / "volumes" / "Radii_Corrections.npy"
    else:
        rc_path = Path(os.path.abspath(__file__)).parents[2] / "library" / "volumes" / "Vis_Radii_Corrections.npy"
    rc_path = str(rc_path)

    # Build function
    def build(resolution):
        if verbose:
            print("Generating new correction table.")
        _ = table_generation(size=3)  # Make sure the fxn is compiled
        LUT = table_generation(resolution, size)
        np.save(rc_path, LUT)

        if verbose:
            print("Table generation complete.")
        return LUT

    if new_build or not os.path.exists(rc_path):
        if verbose:
            print("New build initiated.")
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
