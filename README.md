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
