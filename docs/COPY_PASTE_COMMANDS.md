# Copy-paste commands for CBCT/PlanCT CycleGAN workflows

This page contains command blocks that can be copied directly. Replace only the paths and checkpoint epoch when your folders differ.

## Workflow A: MRI + CBCT -> sCT with PlanCT as the supervision target

Use this when each case has `MRI/`, `CBCT/`, and `PlanCT/`, and `PlanCT` is the target image used for supervision.

Expected raw layout:

```text
raw_cbct_planct/
  train/Case001/{MRI,CBCT,PlanCT}/
  validation/Case101/{MRI,CBCT,PlanCT}/
  test/proceeding/Case201/{MRI,CBCT,PlanCT}/
```

### 1. Register MRI and CBCT to the PlanCT grid

```bash
python tools/cyclegan_cbct_planct.py register-all \
  --raw_root ./raw_cbct_planct \
  --registered_root ./registered_mri_cbct_planct \
  --fixed_name PlanCT \
  --moving_modalities MRI,CBCT \
  --transform rigid
```

### 2. Train with two input channels and one PlanCT target channel

Do not run `prepare-all` for this variant. The registration output already has synchronized filenames under `train/`, `validation/`, and `test/proceeding/`, and there is no separate `CT/` folder for the CBCT+PlanCT preparation script to consume.

```bash
python tools/cyclegan_cbct_planct.py train \
  --data_root ./registered_mri_cbct_planct \
  --runs_root ./runs/CycleMRI_CBCT_sCT_PlanCTTarget \
  --input_modalities MRI,CBCT \
  --target_name PlanCT \
  --use_planct_completion false \
  --batch_size 4 \
  --num_epochs 500 \
  --num_epochs_decay 200
```

### 3. Test / generate sCT

Replace `500` with the checkpoint epoch you want to load.

```bash
python tools/cyclegan_cbct_planct.py test \
  --data_root ./registered_mri_cbct_planct \
  --runs_root ./runs/CycleMRI_CBCT_sCT_PlanCTTarget \
  --input_modalities MRI,CBCT \
  --target_name PlanCT \
  --use_planct_completion false \
  --test_epochs 500
```

## Workflow B: CBCT + PlanCT completion -> sCT with CT as the target

Use this when each case has limited-FOV `CBCT/`, full-grid `PlanCT/`, and target/reference `CT/`.

Expected raw layout:

```text
raw_cbct_planct/
  train/Case001/{CBCT,PlanCT,CT}/
  validation/Case101/{CBCT,PlanCT,CT}/
  test/proceeding/Case201/{CBCT,PlanCT,CT}/
```

### 1. Register CBCT and CT to the PlanCT grid

```bash
python tools/cyclegan_cbct_planct.py register-all \
  --raw_root ./raw_cbct_planct \
  --registered_root ./registered_cbct_planct_ct \
  --fixed_name PlanCT \
  --moving_modalities CBCT,CT \
  --transform rigid
```

### 2. Prepare synchronized full-z-range CycleGAN folders

```bash
python tools/cyclegan_cbct_planct.py prepare-all \
  --raw_root ./registered_cbct_planct_ct \
  --data_root ./data/CBCT_CycleGAN \
  --key_mode auto
```

### 3. Train with PlanCT completion enabled

```bash
python tools/cyclegan_cbct_planct.py train \
  --data_root ./data/CBCT_CycleGAN \
  --runs_root ./runs/CycleCBCT_PlanCT \
  --input_modalities CBCT \
  --target_name CT \
  --use_planct_completion true \
  --planct_name PlanCT \
  --fov_mask_mode nonzero \
  --batch_size 4 \
  --num_epochs 500 \
  --num_epochs_decay 200
```

### 4. Test / generate sCT

Replace `500` with the checkpoint epoch you want to load.

```bash
python tools/cyclegan_cbct_planct.py test \
  --data_root ./data/CBCT_CycleGAN \
  --runs_root ./runs/CycleCBCT_PlanCT \
  --input_modalities CBCT \
  --target_name CT \
  --use_planct_completion true \
  --planct_name PlanCT \
  --fov_mask_mode nonzero \
  --test_epochs 500
```

## If the data is already registered and synchronized

### MRI + CBCT -> sCT, PlanCT target

If your cases already contain same-named slices under `MRI/`, `CBCT/`, and `PlanCT/`, point `--data_root` directly at that root:

```bash
python tools/cyclegan_cbct_planct.py train \
  --data_root ./registered_mri_cbct_planct \
  --runs_root ./runs/CycleMRI_CBCT_sCT_PlanCTTarget \
  --input_modalities MRI,CBCT \
  --target_name PlanCT \
  --use_planct_completion false \
  --batch_size 4 \
  --num_epochs 500 \
  --num_epochs_decay 200
```

```bash
python tools/cyclegan_cbct_planct.py test \
  --data_root ./registered_mri_cbct_planct \
  --runs_root ./runs/CycleMRI_CBCT_sCT_PlanCTTarget \
  --input_modalities MRI,CBCT \
  --target_name PlanCT \
  --use_planct_completion false \
  --test_epochs 500
```

### CBCT + PlanCT completion -> sCT, CT target

If your cases are already registered but still need full-range synchronized output folders, start from preparation:

```bash
python tools/cyclegan_cbct_planct.py prepare-all \
  --raw_root ./registered_cbct_planct_ct \
  --data_root ./data/CBCT_CycleGAN \
  --key_mode auto
```

```bash
python tools/cyclegan_cbct_planct.py train \
  --data_root ./data/CBCT_CycleGAN \
  --runs_root ./runs/CycleCBCT_PlanCT \
  --input_modalities CBCT \
  --target_name CT \
  --use_planct_completion true \
  --planct_name PlanCT \
  --fov_mask_mode nonzero \
  --batch_size 4 \
  --num_epochs 500 \
  --num_epochs_decay 200
```

```bash
python tools/cyclegan_cbct_planct.py test \
  --data_root ./data/CBCT_CycleGAN \
  --runs_root ./runs/CycleCBCT_PlanCT \
  --input_modalities CBCT \
  --target_name CT \
  --use_planct_completion true \
  --planct_name PlanCT \
  --fov_mask_mode nonzero \
  --test_epochs 500
```

## Preview without running

Add `--dry_run` to any wrapper command to print the delegated command without executing it.

```bash
python tools/cyclegan_cbct_planct.py train \
  --data_root ./registered_mri_cbct_planct \
  --runs_root ./runs/CycleMRI_CBCT_sCT_PlanCTTarget \
  --input_modalities MRI,CBCT \
  --target_name PlanCT \
  --use_planct_completion false \
  --dry_run
```
