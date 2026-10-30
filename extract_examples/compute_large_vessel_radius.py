"""Compute the dataset-wide large_vessel_radius value.

Iterates all vessel segmentations in a directory, extracts the graph for each,
and reports the median of per-graph median radii. Use the printed value as the
static large_vessel_radius in extract_directory_static_*.py scripts.
"""
import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd

from vvl.utils.GraphInfo import GraphInfo

vesselseg_dir = r"E:\CVD_backup\PLIS\vessel\leg"
"""
n files: 218
PLIS FOOT
min:     0.009173
max:     0.025088
mean:    0.015486
median:  0.015574   <-- use this as large_vessel_radius Foot


PLIS LEG
min:     0.008856
max:     0.019060
mean:    0.012721
median:  0.011714   <-- use this as large_vessel_radius Leg
"""
filter_length = 0.250
prune_length = 0.0
legacy = True
normalize = False

resolution = [0.012, 0.012, 0.003]


def process_single_file(vesselseg_path):
    # layerseg is not needed for radius extraction; pass None to skip loading it
    g_i = GraphInfo(
        vesselseg_path, None,
        resolution=resolution, filter_length=filter_length,
        prune_length=prune_length, legacy=legacy,
        output_dir=None, depth=None, normalize=normalize,
    )
    return os.path.basename(vesselseg_path), g_i.extract_radius()


if __name__ == "__main__":
    results_folder = os.path.join(vesselseg_dir, "vesselvio")
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    segs = sorted(seg for seg in os.listdir(vesselseg_dir) if ".nii.gz" in seg)
    seg_paths = [os.path.join(vesselseg_dir, seg) for seg in segs]

    print(f"Computing per-graph median radii for {len(seg_paths)} files...")
    with Pool(processes=6, maxtasksperchild=1) as pool:
        results = list(tqdm(pool.imap(process_single_file, seg_paths), total=len(seg_paths)))

    names = [r[0] for r in results]
    radii = np.array([r[1] for r in results], dtype=float)

    df = pd.DataFrame({"name": names, "median_radius": radii})
    out_csv = os.path.join(results_folder, "per_file_median_radii.csv")
    df.to_csv(out_csv, index=False)

    dataset_median = float(np.median(radii))
    dataset_mean = float(np.mean(radii))

    print()
    print(f"Saved per-file radii to: {out_csv}")
    print(f"n files: {len(radii)}")
    print(f"min:     {radii.min():.6f}")
    print(f"max:     {radii.max():.6f}")
    print(f"mean:    {dataset_mean:.6f}")
    print(f"median:  {dataset_median:.6f}   <-- use this as large_vessel_radius")
