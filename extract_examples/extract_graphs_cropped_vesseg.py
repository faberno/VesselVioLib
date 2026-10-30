"""Extract whole vessel graphs from VesNet segmentations, like ``extract_all_graphs.py``.

Inputs are the outputs of ``rsom_vesseg_vesnet_folder.py`` / ``rsom_vesseg_vesnet_sr.py`` (rsom-toolbox root):
``<vessel_dir>/<stem>.nii.gz``, uint8 vessel masks in recon axis order (a, b, z), cropped to the dermis band.

No layer segmentation is used: it only serves the upper/lower split in ``GraphInfo``, which is not wanted
here, and graph extraction itself works on the vessel mask alone (``layerseg_path=None``).

Outputs per vessel dir: ``<vessel_dir>/vesselvio/<stem>.pkl`` (igraph, as ``save_graph`` writes it) and
``<vessel_dir>/vesselvio/graph_summary.csv``.
"""

import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vvl.utils.GraphInfo import GraphInfo

# ---------------------------------------------------------------------------- configuration
STUDY_DIR = Path(r"E:\sr_data\532\532_raw_PatrickJoshuaNew_forTesting\selection")
VESSEL_DIRS = [
    STUDY_DIR / "original" / "vessel_segmentation_vesnet_common",
    STUDY_DIR / "9x" / "interpolated" / "OSRecon" / "vessel_segmentation_vesnet",
    STUDY_DIR / "16x" / "interpolated" / "OSRecon" / "vessel_segmentation_vesnet",
    STUDY_DIR / "25x" / "interpolated" / "OSRecon" / "vessel_segmentation_vesnet",
]
RESULTS_SUBDIR = "vesselvio"

filter_length = 0.250  # remove paths with a length less than this
prune_length = 0  # remove connected endpoint vessels with length less than this
normalize = False  # vessel signal is already cropped to the dermis band
resolution = [0.012, 0.012, 0.003]  # mm, axis order (a, b, z)

SKIP_EXISTING = False
N_WORKERS = 6  # each worker holds one volume plus its skeleton; lower this if RAM runs out
# ------------------------------------------------------------------------------------------


def find_jobs() -> list[tuple[str, str]]:
    jobs = []
    for vessel_dir in VESSEL_DIRS:
        results = vessel_dir / RESULTS_SUBDIR
        for seg in sorted(vessel_dir.glob("*.nii.gz")):
            if seg.name.endswith("_v_rgb.nii.gz"):
                continue
            if SKIP_EXISTING and (results / seg.name.replace(".nii.gz", ".pkl")).is_file():
                continue
            jobs.append((str(seg), str(results)))
    return jobs


def extract_one(job: tuple[str, str]) -> dict:
    vessel_path, results = job
    row = {"source_dir": str(Path(vessel_path).parent), "name": Path(vessel_path).name[: -len(".nii.gz")]}
    t0 = time.perf_counter()
    try:
        Path(results).mkdir(parents=True, exist_ok=True)
        g_i = GraphInfo(
            vessel_path, None,
            resolution=resolution, filter_length=filter_length, prune_length=prune_length,
            legacy=False, output_dir=results, normalize=normalize, save_pickle=True,
        )
        g_i.extract_graph()
        row.update(n_nodes=g_i.i_graph.vcount(), n_edges=g_i.i_graph.ecount(), error="")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def write_summaries(rows: list[dict]):
    df = pd.DataFrame(rows)
    for source_dir, part in df.groupby("source_dir"):
        csv = Path(source_dir) / RESULTS_SUBDIR / "graph_summary.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        if csv.is_file():  # keep rows of earlier runs that were skipped this time
            old = pd.read_csv(csv)
            old = old[[c for c in old.columns if c in part.columns]]
            part = pd.concat([old[~old["name"].isin(part["name"])], part])
        part.sort_values("name").to_csv(csv, index=False)


if __name__ == "__main__":
    jobs = find_jobs()
    print(f"{len(jobs)} vessel masks to process in {len(VESSEL_DIRS)} folders")
    if not jobs:
        raise SystemExit(0)

    with Pool(N_WORKERS) as pool:
        rows = list(tqdm(pool.imap_unordered(extract_one, jobs), total=len(jobs)))
    write_summaries(rows)

    failed = [r for r in rows if r["error"]]
    print(f"\nFinished: {len(rows) - len(failed)} ok, {len(failed)} failed")
    for r in failed:
        print(f"    FAILED: {r['source_dir']} / {r['name']}: {r['error']}")
