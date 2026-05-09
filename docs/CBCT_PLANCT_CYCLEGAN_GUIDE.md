# CBCT + PlanCT CycleGAN quick guide

This guide is the recommended entry point for generating full-pelvis sCT from limited-z-range CBCT using PlanCT completion. For copy-paste-only command blocks, see `docs/COPY_PASTE_COMMANDS.md`.

## What the code does

- `tools/prepare_cbct_planct_cyclegan.py` aligns raw `CBCT/`, `PlanCT/`, and `CT/` DICOM series by `InstanceNumber` or z-position, keeps the full `PlanCT ∩ CT` z-range, and copies CBCT files only where CBCT exists.
- `CycleGAN/data_loader_aug.py` reads the prepared folders. With PlanCT completion enabled, each training input is a single-channel image: CBCT inside valid CBCT FOV and PlanCT outside it; if a z-slice has no CBCT file, PlanCT is used as the full input.
- `CycleGAN/main_CBCT.py` is the CBCT->sCT train/test entrypoint.
- `CycleGAN/solver_val.py` builds the CycleGAN models, trains generators/discriminators, validates, tests, writes reports, and saves generated images/DICOM outputs.
- `tools/cyclegan_cbct_planct.py` is a convenience wrapper that runs preparation, training, and testing commands with consistent paths.

## Required data before preparation

For every patient/case, prepare registered and resampled DICOM folders:

```text
raw_cbct_planct/
  train/
    Case001/
      CBCT/      # may cover only part of the pelvis in z
      PlanCT/    # full pelvis, registered/resampled to the target grid
      CT/        # full pelvis training target/reference
  validation/
    Case101/
      CBCT/
      PlanCT/
      CT/
  test/
    proceeding/
      Case201/
        CBCT/
        PlanCT/
        CT/
```

Important: registration/resampling must happen before using this repository. The convenience wrapper now includes a registration step for MRI/CBCT-to-PlanCT preprocessing, but you should still visually inspect the output and prefer a clinically validated registration workflow when available.

## Register MRI/CBCT to PlanCT before preparation

For MRI+CBCT or CBCT+PlanCT workflows, first resample moving modalities to the PlanCT grid. This improves pixel correspondence before the model sees multi-channel inputs and reduces anatomy leakage from misaligned PlanCT completion.

Register all train/validation/test splits:

```bash
python tools/cyclegan_cbct_planct.py register-all \
  --raw_root ./raw_cbct_planct \
  --registered_root ./registered_cbct_planct \
  --fixed_name PlanCT \
  --moving_modalities MRI,CBCT \
  --transform rigid
```

Then prepare the registered folders:

```bash
python tools/cyclegan_cbct_planct.py prepare-all \
  --raw_root ./registered_cbct_planct \
  --data_root ./data/CBCT_CycleGAN \
  --key_mode auto
```

Notes:

- `--transform rigid` is the default and safest first choice; use `affine` only if scanner geometry/scaling differences justify it.
- `--transform none` only resamples using the existing DICOM geometry and assumes the series are already aligned.
- MRI/CBCT registration can fail when FOV overlap is small or anatomy differs; inspect overlays before training.
- The registration tool writes synchronized DICOM names (`0001.dcm`, `0002.dcm`, ...), so downstream pairing is simpler.

## One-command preparation

```bash
python tools/cyclegan_cbct_planct.py prepare-all \
  --raw_root ./raw_cbct_planct \
  --data_root ./data/CBCT_CycleGAN \
  --key_mode auto
```

Prepared output:

```text
data/CBCT_CycleGAN/
  train/Case001/{CBCT,PlanCT,CT}/
  validation/Case101/{CBCT,PlanCT,CT}/
  test/proceeding/Case201/{CBCT,PlanCT,CT}/
```

`PlanCT/` and `CT/` contain the full pelvis z-range. `CBCT/` contains only covered slices. Missing CBCT slices are intentional.

## Train

```bash
python tools/cyclegan_cbct_planct.py train \
  --data_root ./data/CBCT_CycleGAN \
  --runs_root ./runs/CycleCBCT_PlanCT \
  --batch_size 4 \
  --num_epochs 500 \
  --num_epochs_decay 200 \
  --fov_mask_mode nonzero
```

Equivalent direct command:

```bash
python CycleGAN/main_CBCT.py \
  --mode train \
  --itemA CBCT \
  --train_dir ./data/CBCT_CycleGAN/train \
  --val_dir ./data/CBCT_CycleGAN/validation \
  --log_dir ./runs/CycleCBCT_PlanCT/logs \
  --model_save_dir ./runs/CycleCBCT_PlanCT/models \
  --sample_dir ./runs/CycleCBCT_PlanCT/samples \
  --val_result_dir ./runs/CycleCBCT_PlanCT/val \
  --report_dir ./runs/CycleCBCT_PlanCT/report \
  --use_planct_completion true \
  --planct_name PlanCT \
  --fov_mask_mode nonzero
```

## Test / generate sCT

Replace `500` with the checkpoint epoch you want to load:

```bash
python tools/cyclegan_cbct_planct.py test \
  --data_root ./data/CBCT_CycleGAN \
  --runs_root ./runs/CycleCBCT_PlanCT \
  --test_epochs 500 \
  --fov_mask_mode nonzero
```

Equivalent direct command:

```bash
python CycleGAN/main_CBCT.py \
  --mode test \
  --itemA CBCT \
  --test_epochs 500 \
  --train_dir ./data/CBCT_CycleGAN/train \
  --val_dir ./data/CBCT_CycleGAN/validation \
  --test_dir ./data/CBCT_CycleGAN/test/proceeding \
  --model_save_dir ./runs/CycleCBCT_PlanCT/models \
  --result_dir ./runs/CycleCBCT_PlanCT/results \
  --report_dir ./runs/CycleCBCT_PlanCT/report \
  --use_planct_completion true \
  --planct_name PlanCT \
  --fov_mask_mode nonzero
```

Outputs are written under `runs/CycleCBCT_PlanCT/results` and reports under `runs/CycleCBCT_PlanCT/report`.

## Practical quality checklist

- Register and resample `CBCT`, `PlanCT`, and target `CT` to the same voxel grid.
- Verify `PlanCT` and target `CT` cover the complete pelvis z-range.
- Use patient-level splits, not slice-level splits.
- Visually inspect completed inputs before training.
- Avoid using the exact target CT as PlanCT input for validation/test, or metrics can be artificially optimistic on PlanCT-only z-slices.
- Choose `--fov_mask_mode nonzero` if missing CBCT pixels are zero-padded; choose `non_air` if missing regions are near air HU.

## MRI + CBCT input for sCT with PlanCT supervision

If a registered MRI is available and you want to train `MRI + CBCT -> sCT` using `PlanCT` only as the supervision target, prepare each case with same-named DICOM slices across `MRI/`, `CBCT/`, and `PlanCT/`. If you use the wrapper registration command below, the registered output root already has the required split structure and can be used directly as `--data_root`:

```text
registered_mri_cbct_planct/
  train/
    Case001/
      MRI/      # registered/resampled MRI channel
        0001.dcm
      CBCT/     # registered/resampled CBCT channel
        0001.dcm
      PlanCT/   # target/supervision CT-like image
        0001.dcm
  validation/
    Case101/{MRI,CBCT,PlanCT}/
  test/proceeding/
    Case201/{MRI,CBCT,PlanCT}/
```

Register raw MRI and CBCT to PlanCT first if needed:

```bash
python tools/cyclegan_cbct_planct.py register-all \
  --raw_root ./raw_cbct_planct \
  --registered_root ./registered_mri_cbct_planct \
  --fixed_name PlanCT \
  --moving_modalities MRI,CBCT \
  --transform rigid
```

Do not run `prepare-all` for this variant unless you also have a separate `CT/` folder. The CBCT+PlanCT preparation script is designed for `{CBCT,PlanCT,CT}` cases, while this two-channel variant uses `{MRI,CBCT,PlanCT}` directly.

Run training with two input channels; the model output is sCT and `PlanCT` is the target:

```bash
python tools/cyclegan_cbct_planct.py train \
  --data_root ./registered_mri_cbct_planct \
  --runs_root ./runs/CycleMRI_CBCT_sCT_PlanCTTarget \
  --input_modalities MRI,CBCT \
  --target_name PlanCT \
  --use_planct_completion false \
  --batch_size 4
```

Run testing/generation with the same channel/target configuration:

```bash
python tools/cyclegan_cbct_planct.py test \
  --data_root ./registered_mri_cbct_planct \
  --runs_root ./runs/CycleMRI_CBCT_sCT_PlanCTTarget \
  --input_modalities MRI,CBCT \
  --target_name PlanCT \
  --use_planct_completion false \
  --test_epochs 500
```

Notes:

- This is a two-channel input: channel 1 is MRI, channel 2 is CBCT. The generator output is the one-channel sCT; PlanCT is only the target used to supervise that output.
- `PlanCT` is used as the training target in this setup; it is not the generated product and is not used for CBCT completion. Keep `--use_planct_completion false` to avoid leaking the target into the input.
- MRI, CBCT, and PlanCT must be registered and resampled to the same voxel grid before training.
- If CBCT does not cover the full z-range but MRI and PlanCT do, either crop training to the common MRI/CBCT/PlanCT range or add a separate missing-CBCT handling strategy before using two-channel training.
