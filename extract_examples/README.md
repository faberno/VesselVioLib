# RSOM Feature Extraction — Output Reference

This folder contains example pipelines that turn RSOM (Raster-Scan Optoacoustic
Mesoscopy) scans of skin into tabular **vessel-morphology** and **intensity**
features. The features are designed for downstream group comparisons (e.g.
diabetes vs. control, aging, inflammation), so most of them are summary
statistics over the whole vascular network or over an anatomical sub-region.

Two scripts produce the tables documented here:

| Script | Output files | What it describes |
|---|---|---|
| `extract_directory_static_LargeRadius_upperLower_and_whole_lowmem.py` | `features_whole.csv/.xlsx`, `features_upperLower.csv/.xlsx` | Graph / morphology features of the vessel segmentation |
| `intensity_feats.py` | `features_intensity.csv/.xlsx` | Optoacoustic signal-amplitude features inside the epidermis and vessel masks |
| `layerseg_geometry_feats.py` | `features_layerseg_geometry.csv/.xlsx` | Fast epidermis thickness, transducer distance, and surface angle extraction requiring only layer segmentations |

Every row is one scan, keyed by a `name` column.

---

## 1. Conventions you need to read the tables

### Inputs
- **Vessel segmentation** (`vessel/...`, NIfTI): binary mask of the vasculature.
  It is converted into a skeleton graph (nodes = junctions/endpoints, edges =
  vessel segments) with per-edge `radius_avg`, `length`, `volume`, `tortuosity`.
- **Epidermis / layer segmentation** (`epidermis/...`, NIfTI): binary mask of the
  epidermal layer. Used to define the skin surface and the upper/lower split.
- **Reconstruction volumes** (`recon/*LF.mat`, `*HF.mat`): the optoacoustic
  amplitude images. **LF** = low-frequency band (larger / deeper vessels), **HF**
  = high-frequency band (fine / superficial vessels).

### Physical units
Voxel `resolution = [0.012, 0.012, 0.003]` **mm** → ~12 µm lateral (x, y) and
~3 µm axial (z). Consequently, unless a column name says otherwise:

- **Lengths / radii** are in **mm** (multiply by 1000 for µm).
- **Volumes** are in **mm³**.
- Columns whose name ends in `[um]` are already in **µm**.
- Counts, fractions, ratios, exponents, tortuosity, fractal dimension and graph
  density are **dimensionless**.

### Normalization (important)
Many morphology features are internally divided by the imaging volume
(`/ total_volume`) so they read as **densities per mm³**. The example script sets
`normalize = False` because the input volumes are already cropped to a
**standardized field of view**; in that mode `total_volume = 1.0`, so these
columns are effectively **absolute totals / counts over the standardized
volume** rather than true per-mm³ densities. Either way the values are
comparable *across scans* as long as the cropping is consistent. Columns marked
"density" below carry this caveat.

### Small vs. large vessels
A single radius threshold splits the network into "small" and "large" vessels.
In the example script this is a **dataset-wide static median radius**
(`large_vessel_radius`, e.g. `0.015574 mm` ≈ 15.6 µm for PLIS foot scans), chosen
so the split is consistent across the whole cohort. Many features are reported
three times: overall, `..._large_vessel_...`, and `..._small_vessel_...`.

### Whole vs. upper/lower
- `features_whole.*` — one set of features over the entire vessel network.
- `features_upperLower.*` — the **same features computed separately** for the
  superficial and deep compartments, with `_upper` / `_lower` suffixes, plus an
  `upper_lower_depth` column.
  - `upper_lower_depth` — axial depth **in voxels** below the epidermal surface
    where the split is made (×3 µm for depth in µm; e.g. 40 → 120 µm). Either
    fixed (`vp_depth`) or auto-detected from the vessel depth histogram.
  - **upper** ≈ the superficial / dermal capillary plexus; **lower** ≈ the deeper
    reticular vessels.

### Experimental columns
Feature keys containing `experimental` (e.g. connected-component metrics,
degree assortativity, unnormalized cycle lengths) are **dropped before writing**
and will not appear in the CSVs.

---

## 2. Vessel / morphology features (`features_whole`, `features_upperLower`)

> In `features_upperLower` each feature appears twice with `_upper` / `_lower`
> suffixes. Units below are for `normalize=False` (absolute over the
> standardized FOV); divide by mm³ for the normalized interpretation.

### Overall architecture
| Column | Unit | Interpretation |
|---|---|---|
| `fractal_dimension` | – | Box-counting fractal dimension of the 3D vessel mask. Higher = more space-filling, structurally complex branching. |
| `vascular_area_fraction` | – (0–1) | Fraction of the XY field of view covered by the top-down vessel projection (max-intensity projection). A 2D "vessel area density"; sensitive to overall perfusion coverage. |
| `density` | – | Graph density = edges / maximum possible edges. How interconnected the network graph is. |

### Blood volume
| Column | Unit | Interpretation |
|---|---|---|
| `total_blood_volume` | mm³ (density) | Total intravascular volume of all segments. Overall blood/tissue volume. |
| `large_vessel_blood_volume` | mm³ (density) | Volume contributed by vessels ≥ threshold radius. |
| `small_vessel_blood_volume` | mm³ (density) | Volume contributed by vessels < threshold radius (microvasculature). |

### Length
| Column | Unit | Interpretation |
|---|---|---|
| `total_vessel_length` | mm (density) | Summed length of all vessel segments — total network length / length density. |
| `total_large_vessel_length` | mm (density) | Summed length of large vessels. |
| `total_small_vessel_length` | mm (density) | Summed length of small vessels (capillary-bed extent). |
| `median_vessel_length` | mm | Typical segment length (not normalized). |
| `median_large_vessel_length` | mm | Typical large-vessel segment length. |
| `median_small_vessel_length` | mm | Typical small-vessel segment length. |

### Radius
| Column | Unit | Interpretation |
|---|---|---|
| `median_radius` | mm | Typical vessel radius. Vessel caliber; shifts with dilation/constriction or remodeling. |
| `median_large_vessel_radius` | mm | Typical caliber among large vessels. |
| `median_small_vessel_radius` | mm | Typical caliber among small vessels. |
| `dominant_vessel_radius` | mm | Length-weighted radius of the single thickest traced vessel. |
| `dominant_vessel_length` | mm | Total length of that dominant (thickest) vessel path. |

### Branching / junctions
| Column | Unit | Interpretation |
|---|---|---|
| `#bifurcations` | count (density) | Number of graph nodes with degree > 2 — branch points. Branching richness. |
| `num_junctions` | count (density) | Skeleton junction count (skan). |
| `branch_number` | count (density) | Number of skeleton branches (skan, branch-distance > 9). |
| `branch_j2e_total` | count (density) | Junction-to-endpoint branches (terminal twigs). |
| `branch_j2j_total` | count (density) | Junction-to-junction branches (connecting segments). |
| `terminal_vessel_density` | count (density) | Number of vessel endpoints (degree-1 leaf nodes). High = many fine tips reaching into tissue. |
| `mean_degree` | – | Mean node degree of the network. |
| `mean_large_vessel_degree` | – | Mean degree of nodes touching a large vessel. |
| `mean_small_vessel_degree` | – | Mean degree of nodes touching a small vessel. |

### Bifurcation geometry (Murray's law)
| Column | Unit | Interpretation |
|---|---|---|
| `median_bifurcation_exponent` | – | Murray's-law exponent fit at bifurcations (parent vs. children radii). ~3 is the theoretical optimum for efficient laminar flow; deviation can signal remodeling. |
| `median_large_vessel_bifurcation_exponent` | – | Same, at bifurcations whose parent is a large vessel. |
| `median_small_vessel_bifurcation_exponent` | – | Same, parent is a small vessel. |
| `median_branch_length_ratio` | – | Parent branch length / radius at bifurcations. Slenderness of branches. |
| `median_large_vessel_branch_length_ratio` | – | Same, large-vessel parents. |
| `median_small_vessel_branch_length_ratio` | – | Same, small-vessel parents. |

### Tortuosity (note: keys say `mean_` but the value is the median)
| Column | Unit | Interpretation |
|---|---|---|
| `mean_tortuosity` | – (≥1) | Median path length / straight-line distance of segments. >1 = winding/twisted vessels (often raised in inflammation, diabetic microangiopathy). |
| `mean_large_vessel_tortuosity` | – | Median tortuosity of large vessels. |
| `mean_small_vessel_tortuosity` | – | Median tortuosity of small vessels. |

### Loops / cycles
| Column | Unit | Interpretation |
|---|---|---|
| `#cycles` | count (density) | Number of independent loops (cycle basis). Vascular loop/anastomosis density. |
| `median_cycle_length` | mm (density) | Median loop perimeter length. |
| `max_cycle_length` | mm (density) | Largest loop perimeter length. |

### Vessel counts (greedy thickest-path tracing)
| Column | Unit | Interpretation |
|---|---|---|
| `vessel_count` | count | Number of distinct vessels, traced by greedily following the thickest neighbor from each seed edge until a tip. |
| `large_vessel_count` | count | Traced vessels whose length-weighted radius ≥ threshold. |
| `small_vessel_count` | count | Traced vessels whose length-weighted radius < threshold. |

---

## 3. Intensity features (`features_intensity`)

These come from `intensity_feats.py`. The optoacoustic amplitude (LF and HF
bands) is sampled inside the **epidermis mask** and inside the **upper/lower
vessel masks**. Amplitudes are clipped to ≥ 0 and reported in **arbitrary units
(a.u.)** — they scale with optical absorption (hemoglobin / melanin content) and
are sensitive to acquisition gain, so prefer relative comparisons within a
consistently processed cohort. `_hf` = high-frequency band (fine vessels), `_lf`
= low-frequency band (larger/deeper vessels).

### Epidermis (layer segmentation)
| Column | Unit | Interpretation |
|---|---|---|
| `layseg_int_max_hf` | a.u. | Peak HF amplitude in the epidermis (melanin-dominated). |
| `layseg_int_mean_hf` | a.u. | Mean HF amplitude in the epidermis. |
| `layseg_int_max_lf` | a.u. | Peak LF amplitude in the epidermis. |
| `layseg_int_mean_lf` | a.u. | Mean LF amplitude in the epidermis. |
| `layseg_thickness [um]` | µm | Mean epidermal thickness (mean voxel depth-extent × 3 µm). Thickens in some pathologies. |
| `dist_transducer [um]` | µm | Mean depth of the epidermis surface from the top of the volume (× 3 µm). Probe coupling / surface position. |
| `surface_angle` | degrees | Tilt of a plane fitted to the skin surface. Quality/positioning metric (how oblique the skin sits relative to the imaging plane). |

### Vessels — reported for both `_upper` and `_lower`
| Column | Unit | Interpretation |
|---|---|---|
| `vesseg_int_max_hf_upper` / `_lower` | a.u. | Peak HF amplitude inside the upper / lower vessel mask. |
| `vesseg_int_mean_hf_upper` / `_lower` | a.u. | Mean HF amplitude inside the upper / lower vessel mask. |
| `vesseg_int_max_lf_upper` / `_lower` | a.u. | Peak LF amplitude inside the upper / lower vessel mask. |
| `vesseg_int_mean_lf_upper` / `_lower` | a.u. | Mean LF amplitude inside the upper / lower vessel mask. |

Vessel intensity acts as a proxy for blood/hemoglobin absorption within the
segmented vasculature; splitting by upper/lower and by LF/HF separates the
superficial fine plexus from the deeper larger vessels.

---

## Notes & caveats
- The `_upper`/`_lower` vessel masks for intensity are re-aligned to the recon
  by shifting them to the epidermal surface depth (`upper_lower_depth` from the
  morphology run is reused), so the two pipelines must be run in order:
  morphology first (it writes `features_upperLower.csv` and the
  `*_upper.nii.gz` / `*_lower.nii.gz` masks), then intensity.
- Intensity amplitudes (`*_hf`, `*_lf`) are uncalibrated a.u.; the two `###`
  comments in the source flag that the absolute scale is large and unit-less.
  Treat them as relative, not absolute, measures.
- A vessel-radius comment header in the extraction script records the per-cohort
  median radii used to pick `large_vessel_radius`.
