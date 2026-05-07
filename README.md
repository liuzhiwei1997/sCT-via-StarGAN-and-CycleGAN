# sCT Generation via StarGAN and CycleGAN

This repository contains the code used in the paper **"Synthetic CT Generation from CBCT and MRI Using StarGAN in the Pelvic Region"**. The work explores the application of StarGAN for generating synthetic CT (sCT) images from CBCT and MRI scans compare to the use of CycleGAN, specifically in the pelvic region.

**Published in:** *Radiation Oncology*
**DOI:** [10.1186/s13014-025-02590-2](https://doi.org/10.1186/s13014-025-02590-2)

## Data layout expected by the training scripts

All loaders in this repo read **DICOM (`.dcm`) files** and infer domains from folder names.

### 1) CycleGAN (`CycleGAN/main_MRI.py` and `CycleGAN/main_CBCT.py`)

`PairedDICOMFolder` expects paired slices between `itemA` and `CT`, matched by the **same filename** inside a case. Two layouts are supported:

```text
# Case-based (recommended)
<data_root>/train/
  Case001/
    MRI/              # or CBCT when --itemA CBCT
      0001.dcm
      0002.dcm
    CT/
      0001.dcm
      0002.dcm
  Case002/
    MRI/
    CT/

# Flat (single-case)
<data_root>/train/
  MRI/
    0001.dcm
  CT/
    0001.dcm
```

Example (MRI→CT):

```bash
python CycleGAN/main_MRI.py \
  --itemA MRI \
  --train_dir ./data/MRI_CycleGAN/train
```

Example (CBCT→CT):

```bash
python CycleGAN/main_CBCT.py \
  --itemA CBCT \
  --train_dir ./data/CBCT_CycleGAN/train
```

### 2) StarGAN (`StarGAN/main_transfer.py`, `StarGAN/main_MR_only.py`)

`DICOMFolder` reads one directory per domain/class. Folder name becomes class label index.

```text
<data_root>/train_transfer/
  CT/
    ct_0001.dcm
    ct_0002.dcm
  CBCT/
    cbct_0001.dcm
    cbct_0002.dcm
  MRI/
    mr_0001.dcm
    mr_0002.dcm
```

For MR-only pretrain, commonly keep only `MRI` and `CT` domains:

```text
<data_root>/train_mr_only/
  MRI/
    mr_0001.dcm
  CT/
    ct_0001.dcm
```

### 3) Practical checklist before training

- Use `.dcm` suffix for all slices.
- For CycleGAN pairing, ensure `itemA` and `CT` have identical filenames per case.
- Ensure DICOM contains `RescaleSlope` and `RescaleIntercept` (used in preprocessing).
- Keep 2D slice size consistent or let script resize via `--image_size`.

## Solve MRI/CT filename and slice-count mismatch (CycleGAN)

CycleGAN pairing is based on **exact same filenames** between `itemA` and `CT`. If your MRI and CT names/counts differ, use:

```bash
python tools/prepare_paired_dicom.py \
  --mri_dir /path/to/raw/MRI \
  --ct_dir /path/to/raw/CT \
  --output_case_dir ./data/MRI_CycleGAN/train/case001 \
  --key_mode auto
```

What this script does:
- reads DICOM headers from MRI and CT series
- matches slices by `InstanceNumber` (or by `ImagePositionPatient` z when needed)
- keeps only the intersection
- writes synchronized names (`0001.dcm`, `0002.dcm`, ...) under:
  - `case001/MRI/`
  - `case001/CT/`

Dry-run example:

```bash
python tools/prepare_paired_dicom.py \
  --mri_dir /path/to/raw/MRI \
  --ct_dir /path/to/raw/CT \
  --output_case_dir ./data/MRI_CycleGAN/train/case001 \
  --dry_run
```

Batch example (process all cases under `train/`):

```bash
python tools/prepare_paired_dicom.py \
  --input_root ./data/MRI_CycleGAN/train \
  --output_root ./data/MRI_CycleGAN/train_paired \
  --key_mode auto
```

This scans all case folders (e.g., `case001`, `case002`) and processes each case that contains both `MRI/` and `CT/` subfolders.

## CBCT limited-FOV completion with PlanCT for CycleGAN

For CBCT-to-sCT training, limited scan range/FOV can leave pelvic anatomy missing from the CBCT input. The CycleGAN CBCT loader can now build a **single-channel completed input** for the full pelvis: slices with CBCT use CBCT pixels inside the detected FOV and PlanCT pixels outside that mask, while z-slices not covered by CBCT use PlanCT as the complete input.

Expected case layout:

```text
<data_root>/train/
  Case001/
    CBCT/
      0002.dcm        # optional per z-slice; only present where CBCT covers the pelvis
    PlanCT/
      0001.dcm
      0002.dcm
      0003.dcm
    CT/
      0001.dcm
      0002.dcm
      0003.dcm
```

Run CBCT CycleGAN with PlanCT completion:

```bash
python CycleGAN/main_CBCT.py \
  --itemA CBCT \
  --train_dir ./data/CBCT_CycleGAN/train \
  --val_dir ./data/CBCT_CycleGAN/validation \
  --use_planct_completion true \
  --planct_name PlanCT \
  --fov_mask_mode nonzero
```

Mask modes:

- `nonzero` (default): use CBCT where the raw CBCT pixel value is not zero; fill zero/padded regions from PlanCT.
- `non_air`: use CBCT where CBCT HU is greater than `--fov_threshold` (default `-950`).
- `all_cbct`: disable in-plane filling for slices that have CBCT; PlanCT-only z-slices still use PlanCT because no CBCT slice exists.

For z-direction completion, the required full output range is defined by matching `PlanCT/` and `CT/` filenames. `CBCT/` files are optional on that range. If `PlanCT/0001.dcm` and `CT/0001.dcm` exist but `CBCT/0001.dcm` does not, the loader uses `PlanCT/0001.dcm` as the generator input and `CT/0001.dcm` as the target.

### Registration and quality checklist before preparing folders

The preparation script only matches slices and renames/copies DICOM files; it **does not** register, resample, crop, or correct image geometry. Before running `tools/prepare_cbct_planct_cyclegan.py`, prepare each case so that `CBCT`, `PlanCT`, and target `CT` are already spatially consistent:

- Register `PlanCT` to the CBCT/target-CT patient coordinate system before completion. Use at least rigid registration; consider deformable registration when anatomy, bladder/rectum filling, couch position, or body contour differs substantially.
- Resample all modalities to the same image grid: same matrix size, pixel spacing, slice spacing/thickness, orientation, origin, and z ordering. The loader combines same-named slices pixel-by-pixel, so mismatched geometry will paste PlanCT anatomy into the wrong CBCT locations.
- Check that `PlanCT/` and target `CT/` cover the full pelvis range you want the sCT to output. `CBCT/` may cover only a subset of those z-slices.
- Keep the target `CT` as the evaluation/training reference and `PlanCT` as input prior information. If `PlanCT` is exactly the same DICOM series as the target `CT`, metrics can be artificially optimistic on PlanCT-only z-slices because the input already contains the target anatomy.
- Apply consistent preprocessing before export: artifact correction if available, body/FOV mask review, HU calibration/rescale tags (`RescaleSlope`, `RescaleIntercept`), and removal of corrupted or duplicate slices.
- Split train/validation/test by patient, not by slice, to avoid leakage across neighboring slices from the same patient.
- Inspect several completed inputs visually after preparation: CBCT-covered slices should show CBCT inside the FOV and PlanCT outside; CBCT-missing superior/inferior slices should be complete PlanCT inputs aligned with target CT.

### Prepare simple CBCT/PlanCT/CT CycleGAN folders

If raw case folders contain `CBCT/`, `PlanCT/`, and `CT/` series with different filenames or slice counts, use the organizer:

```bash
python tools/prepare_cbct_planct_cyclegan.py \
  --cbct_dir /path/to/raw/Case001/CBCT \
  --planct_dir /path/to/raw/Case001/PlanCT \
  --ct_dir /path/to/raw/Case001/CT \
  --output_case_dir ./data/CBCT_CycleGAN/train/Case001 \
  --key_mode auto
```

Batch mode:

```bash
python tools/prepare_cbct_planct_cyclegan.py \
  --input_root /path/to/raw/train_cases \
  --output_root ./data/CBCT_CycleGAN/train \
  --key_mode auto
```

The script aligns the series by `InstanceNumber` or `ImagePositionPatient` z-position, keeps the full common `PlanCT`/target `CT` z-range, and writes synchronized names (`0001.dcm`, `0002.dcm`, ...). CBCT files are copied only for z-slices where CBCT exists; missing CBCT files are intentional and cause `CycleGAN/main_CBCT.py` to use PlanCT for those full slices.
