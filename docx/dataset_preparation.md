# Dataset Preparation

This document describes the dataset preparation workflow used by FlowDiT-RM. It is based on the original `data_pre/preprocess_dataset.py` and `data_pre/split_dataset.py` scripts from the project.

## Overview

The preparation pipeline has two stages:

1. Preprocess the raw radio-map data into compact `.npz` sample files.
2. Split the processed samples into train, validation, and test subsets at the map level.

The output of the first stage is used directly by the second stage and by the PyTorch dataset loader.

## Raw Data Layout

The preprocessing script expects a raw dataset root directory with at least two subdirectories:

```text
PPData5D-success/
  png/
    .../*_ss_z00.png
    .../*_ss_z01.png
    .../*_ss_z02.png
  npz/
    *_bdtr.npz
```

The PNG files provide the radio-map signal layers. For each sample, the script expects three height layers:

- `z00`: receiver height 1.5 m
- `z01`: receiver height 30.0 m
- `z02`: receiver height 200.0 m

The terrain file is stored as an `.npz` file and is matched by scene id. It must contain `terrain_yx`. If `inBldg_zyx` is present, it is used as the three building-occupancy channels. If it is missing, the script falls back to three zero-valued building channels.

## File Matching

The preprocessing stage scans all files matching:

```text
png/**/*_ss_z00.png
```

For each `z00` file, it derives:

- `sample_id`, for example `T06C0D0002_n00_f00`
- `scene_id`, for example `T06C0D0002_n00`
- matching `z01` and `z02` PNG files
- matching terrain file: `npz/{scene_id}_bdtr.npz`

A sample is processed only when all required files exist.

## Frequency Mapping

The frequency id is parsed from the last field of `sample_id`. The original mapping is:

| Frequency id | Frequency |
| --- | ---: |
| `f00` | 150 MHz |
| `f01` | 1500 MHz |
| `f02` | 1700 MHz |
| `f03` | 3500 MHz |
| `f04` | 22000 MHz |

Samples with an unknown frequency id are skipped.

## Transmitter Detection

For each valid sample, transmitters are detected from the `z00` radio map using:

```python
detect_tx_from_png(
    png_path=sample["z00"],
    percentile=98.0,
    min_distance=10,
)
```

If no transmitter is detected, the sample is treated as invalid and skipped. Exceptions during transmitter detection are also counted as failed samples.

## Spatial Conditions

The spatial condition tensor is built from terrain and building information:

```text
cond_spatial = [terrain_yx, inBldg_zyx]
```

The final shape is:

```text
(4, H, W)
```

The four channels are:

1. Terrain elevation
2. Building occupancy at height layer 1
3. Building occupancy at height layer 2
4. Building occupancy at height layer 3

Samples are skipped if the terrain or building arrays have invalid dimensions, or if the building map spatial size does not match the terrain map.

## Physical Prior Generation

The script generates a three-layer FSPL physical prior with:

```python
generate_3d_fspl_tensor(
    shape=z00_mat.shape,
    coords=coords,
    powers=powers,
    terrain_yx=terrain_yx,
    rx_heights_agl=[1.5, 30.0, 200.0],
    dynamic_range_db=dynamic_gamma,
)
```

The dynamic range is topology-aware. It is computed from the building blockage ratio:

```python
blockage_ratio = np.mean(inBldg_zyx > 0)
dynamic_gamma = 60.0 + (60.0 * blockage_ratio)
```

This gives a lower dynamic range for open areas and a higher dynamic range for dense built environments.

Samples are skipped if the generated FSPL prior contains `NaN` or `Inf`.

## Processed Sample Format

Each valid sample is saved as a compressed `.npz` file:

```text
Dataset5D/{sample_id}_processed.npz
```

Each file contains:

| Key | Shape | Description |
| --- | --- | --- |
| `gt_map` | `(3, H, W)` | Three-layer ground-truth radio map |
| `fspl_prior` | `(3, H, W)` | Three-layer FSPL physical prior |
| `cond_spatial` | `(4, H, W)` | Terrain plus building condition tensor |
| `freq_mhz` | scalar | Frequency in MHz |

The preprocessing script also records global statistics:

- minimum and maximum signal value
- minimum and maximum condition value
- processed sample count
- failed sample count

## Default Preprocessing Paths

The original script uses:

```python
RAW_DATA_DIR = "/workspace/PPData5D-success"
OUTPUT_DIR = "/workspace/Dataset5D"
```

Run preprocessing with:

```bash
python data_pre/preprocess_dataset.py
```

## Dataset Split Strategy

After preprocessing, the split script reads:

```text
/workspace/Dataset5D/*.npz
```

and creates:

```text
/workspace/MapLevel_Split_3Way/
  train/
  val/
  test/
```

The default split ratios are:

| Split | Ratio |
| --- | ---: |
| Train | 0.7 |
| Validation | 0.1 |
| Test | 0.2 |

The random seed is fixed:

```python
random.seed(42)
```

## Map-Level Split

The split is performed inside each scene category. The script parses each processed filename as follows:

- `scene_id = fname[:3]`
- `map_id = fname.split("_")[0]`

All samples belonging to the same `map_id` are assigned to the same subset. This avoids leakage between train, validation, and test sets at the map level.

If a scene has too few maps, the validation split may be set to zero to keep at least one training map.

The split script creates symbolic links instead of copying files:

```python
os.symlink(os.path.abspath(f_path), os.path.join(target_dir, fname))
```

This keeps the split directories lightweight while preserving a single processed data source.

## Default Split Paths

The original script uses:

```python
SOURCE_DIR = "/workspace/Dataset5D"
BASE_SPLIT_DIR = "/workspace/MapLevel_Split_3Way"
```

Run splitting with:

```bash
python data_pre/split_dataset.py
```

## Recommended Workflow

1. Prepare the raw dataset under `PPData5D-success/`.
2. Run `preprocess_dataset.py` to generate `Dataset5D/*.npz`.
3. Review the preprocessing report and confirm the processed and failed sample counts.
4. Run `split_dataset.py` to create the map-level train, validation, and test directories.
5. Use the split directories as training and evaluation inputs.

The training code can then consume:

```text
/workspace/MapLevel_Split_3Way/train
/workspace/MapLevel_Split_3Way/val
/workspace/MapLevel_Split_3Way/test
```
