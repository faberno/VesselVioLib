import numpy as np


def get_surface_angle(surface):
    ### angle is computed as square root of the squared x and y slopes (a and b) of a fitted plane.
    nx, ny = surface.shape
    x, y = np.meshgrid(np.arange(ny), np.arange(nx))# coordinate grid
    X = np.column_stack((x.ravel(), y.ravel(), np.ones(nx*ny)))# flatten
    Zf = surface.ravel()
    a, b, c = np.linalg.lstsq(X, Zf, rcond=None)[0]# least squares plane fit
    theta = np.degrees(np.arctan(np.sqrt(a*a + b*b))) #tilt angle
    return int(theta)
 
def extract_int_features(
    volume_lay,
    volume_upper,
    volume_lower,
    recon,
    upper_lower_depth,
):
    def layseg_int_features(lf, hf, volume_lay):
        surface = np.argmax(volume_lay, axis=2)
        dist_transducer = np.mean(surface)
        surface_angle = get_surface_angle(surface)
 
        layseg_int_max_hf = hf.max()
        layseg_int_mean_hf = hf.mean()
        layseg_int_max_lf = lf.max()
        layseg_int_mean_lf = lf.mean()
 
        depth_counts = np.sum(volume_lay > 0, axis=-1)  # shape: (y, x)
        layseg_thickness = np.mean(depth_counts[depth_counts > 0])
 
        return {
            "layseg_int_max_hf": layseg_int_max_hf,  ###welche einheit, warum values so groß?
            "layseg_int_mean_hf": layseg_int_mean_hf,
            "layseg_int_max_lf": layseg_int_max_lf,
            "layseg_int_mean_lf": layseg_int_mean_lf,
            "layseg_thickness [um]": int(layseg_thickness * 3),
            "dist_transducer [um]": int(dist_transducer * 3),
            "surface_angle": surface_angle,
        }
 
    def vesseg_int_feats(lf, hf):
        vesseg_upper_int_max_hf = hf.max()
        vesseg_upper_int_mean_hf = hf.mean()
        vesseg_upper_int_max_lf = lf.max()
        vesseg_upper_int_mean_lf = lf.mean()
 
        return {
            "vesseg_int_max_hf": vesseg_upper_int_max_hf,  ###welche einheit, warum values so groß?
            "vesseg_int_mean_hf": vesseg_upper_int_mean_hf,
            "vesseg_int_max_lf": vesseg_upper_int_max_lf,
            "vesseg_int_mean_lf": vesseg_upper_int_mean_lf,
        }
   
    recon_lf = recon[0]
    recon_hf = recon[1]
 
    volume_lay = np.array(volume_lay, dtype=np.float32)
    volume_upper = np.array(volume_upper, dtype=np.float32)
    volume_lower = np.array(volume_lower, dtype=np.float32)
 
 
    # recon_hf = np.transpose(recon_hf, axes=(1, 2, 0))
    # recon_lf = np.transpose(recon_lf, axes=(1, 2, 0))
    recon_hf = np.clip(recon_hf, 0, np.max(recon_hf))
    recon_lf = np.clip(recon_lf, 0, np.max(recon_lf))
 
    features = {}
    layseg_hf = recon_hf[volume_lay == 1]
    layseg_lf = recon_lf[volume_lay == 1]
 
    features.update(layseg_int_features(layseg_lf, layseg_hf, volume_lay))
 
    shifted_vesseg_upp = np.zeros(recon_hf.shape, dtype=recon_hf.dtype)
    depth_map_full = volume_lay.shape[2] - np.argmax(volume_lay[..., ::-1], axis=2)
    shift = np.min(depth_map_full)
    shifted_vesseg_upp[:, :, shift : shift + volume_upper.shape[2]] = volume_upper
    vesseg_upper_hf = recon_hf[shifted_vesseg_upp == 1]
    vesseg_upper_lf = recon_lf[shifted_vesseg_upp == 1]
    upper_ves_feats = vesseg_int_feats(vesseg_upper_lf, vesseg_upper_hf)
    upper_ves_feats = {
        k + "_upper": upper_ves_feats[k] for k in upper_ves_feats.keys()
    }
    features.update(upper_ves_feats)
 
    shifted_vesseg_low = np.zeros(recon_hf.shape, dtype=recon_hf.dtype)
    depth_map_full = volume_lay.shape[2] - np.argmax(volume_lay[..., ::-1], axis=2)
    shift = np.min(depth_map_full) + upper_lower_depth
    shifted_vesseg_low[:, :, shift : shift + volume_lower.shape[2]] = volume_lower
    vesseg_lower_hf = recon_hf[shifted_vesseg_low == 1]
    vesseg_lower_lf = recon_lf[shifted_vesseg_low == 1]
    lower_ves_feats = vesseg_int_feats(vesseg_lower_lf, vesseg_lower_hf)
    lower_ves_feats = {k + "_lower": lower_ves_feats[k] for k in lower_ves_feats.keys()}
    features.update(lower_ves_feats)

    return features


def _clean_up_name(name):
    name = name.replace("_processed", "")
    name = name.replace("__l", "")
    name = name.replace("_l", "")
    name = name.replace("_0000", "")
    name = name.replace("_0001", "")
    name = name.replace("_ves.nii.gz", "")
    name = name.replace("_ed.nii.gz", "")
    name = name.replace(".nii.gz", "")
    return name


def _process_one(args):
    """Top-level (picklable) worker for multiprocessing."""
    import nibabel as nib
    from scipy.io import loadmat

    name, layerseg_path, upper_seg_path, lower_seg_path, recon_lf_path, recon_hf_path, depth = args
    try:
        volume_lay = nib.load(layerseg_path).get_fdata()
        volume_upper = nib.load(upper_seg_path).get_fdata()
        volume_lower = nib.load(lower_seg_path).get_fdata()
        recon_lf = np.transpose(loadmat(recon_lf_path)["R"], axes=(1, 0, 2))
        recon_hf = np.transpose(loadmat(recon_hf_path)["R"], axes=(1, 0, 2))

        feats = extract_int_features(
            volume_lay=volume_lay,
            volume_upper=volume_upper,
            volume_lower=volume_lower,
            recon=(recon_lf, recon_hf),
            upper_lower_depth=int(depth),
        )
        return {"name": name, **feats}
    except Exception as e:
        print(f"FAILED {name}: {e!r}")
        return {"name": name}


if __name__ == "__main__":
    # Folder example. Processes every vesselseg in `vesselseg_dir` in parallel.
    import os
    from pathlib import Path
    from multiprocessing import Pool, cpu_count

    import pandas as pd
    from tqdm import tqdm

    vesselseg_dir = r"E:\CVD_backup\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\vessel\arm"
    layerseg_dir = r"E:\CVD_backup\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\epidermis"
    recon_dir = r"\\TUMEBB1-WS1.med.tum.de\data\derived_data\OPTOMICS_1.1-12-2348-EST_UTARTU\rsom\processed\recon"
    results_folder = os.path.join(vesselseg_dir, "vesselvio")

    # upper_lower_depth per scan was written by the main extraction pipeline.
    ul_csv = os.path.join(results_folder, "features_upperLower.csv")
    depth_map = dict(
        zip(*pd.read_csv(ul_csv, usecols=["name", "upper_lower_depth"]).values.T)
    )

    # Index recon + layerseg directories once so the worker doesn't re-scan them.
    layseg_index = {_clean_up_name(f): os.path.join(layerseg_dir, f) for f in os.listdir(layerseg_dir)}
    recon_files = os.listdir(recon_dir)

    args_list = []
    skipped = []
    for seg in os.listdir(vesselseg_dir):
        if not seg.endswith(".nii.gz"):
            continue
        name = seg.replace(".nii.gz", "")
        base = _clean_up_name(seg)

        if name not in depth_map or pd.isna(depth_map[name]):
            skipped.append((name, "no upper_lower_depth row"))
            continue
        upper_seg_path = os.path.join(results_folder, f"{name}_upper.nii.gz")
        lower_seg_path = os.path.join(results_folder, f"{name}_lower.nii.gz")
        if not (os.path.exists(upper_seg_path) and os.path.exists(lower_seg_path)):
            skipped.append((name, "missing upper/lower vesselseg"))
            continue
        layerseg_path = layseg_index.get(base)
        if layerseg_path is None:
            skipped.append((name, "no matching layerseg"))
            continue
        lf = next((f for f in recon_files if f.startswith(base.replace("Seg_","R_")) and f.endswith("LF.mat")), None)
        hf = next((f for f in recon_files if f.startswith(base.replace("Seg_","R_")) and f.endswith("HF.mat")), None)
        if lf is None or hf is None:
            skipped.append((name, "no LF/HF recon"))
            continue

        args_list.append((
            name, layerseg_path, upper_seg_path, lower_seg_path,
            os.path.join(recon_dir, lf), os.path.join(recon_dir, hf),
            depth_map[name],
        ))

    if skipped:
        print(f"Skipping {len(skipped)} scans:")
        for n, r in skipped:
            print(f"  {n}: {r}")

    n_workers = max(1, cpu_count() - 1)
    print(f"Extracting intensity features for {len(args_list)} scans using {n_workers} workers...")
    with Pool(processes=n_workers, maxtasksperchild=1) as pool:
        results = list(tqdm(pool.imap_unordered(_process_one, args_list), total=len(args_list)))

    # results = []
    # for ele in args_list:
    #     results.append(_process_one(ele))

    df = pd.DataFrame(results).sort_values("name").reset_index(drop=True)
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(results_folder, "features_intensity.csv"), index=False)
    df.to_excel(os.path.join(results_folder, "features_intensity.xlsx"), index=False)