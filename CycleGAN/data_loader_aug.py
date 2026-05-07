"""Paired DICOM loader used by the CycleGAN training scripts.

The loader can optionally build a CBCT input that is completed with PlanCT
outside the CBCT field-of-view (FOV). This keeps the CycleGAN model input as a
single channel while giving the generator full-pelvis context.
"""

from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


CT_LIKE_MODALITIES = {"CT", "CBCT", "PLANCT", "PLAN_CT", "PLAN-CT"}


class PairedDICOMFolder(data.Dataset):
    """Load paired modality slices for CycleGAN training.

    Expected layouts:

    * ``root/CASE_ID/<itemA>/*.dcm`` and ``root/CASE_ID/CT/*.dcm``
    * ``root/<itemA>/*.dcm`` and ``root/CT/*.dcm``

    When ``use_planct_completion`` is enabled, each case must also contain a
    ``PlanCT`` folder with filenames matching ``CT``. ``CBCT`` is optional per
    slice: slices with a matching CBCT file use CBCT inside the detected FOV and
    PlanCT outside it, while PlanCT/CT slices without CBCT use PlanCT as the
    complete input. Pairing is performed by filename within each case directory.
    """

    def __init__(
        self,
        root,
        item_a,
        image_size=(512, 512),
        mode="train",
        planct_name="PlanCT",
        use_planct_completion=False,
        fov_mask_mode="nonzero",
        fov_threshold=-950.0,
        input_modalities=None,
        target_name="CT",
    ):
        self.root = Path(root)
        self.item_a = item_a
        self.mode = mode
        self.planct_name = planct_name
        self.input_modalities = list(input_modalities or [item_a])
        self.target_name = target_name
        self.use_planct_completion = use_planct_completion
        self.fov_mask_mode = fov_mask_mode
        self.fov_threshold = fov_threshold
        self.transform = self._build_transform(image_size, mode)
        self.pairs = self._find_pairs()

        if not self.pairs:
            planct_msg = f", '{self.planct_name}'" if self.use_planct_completion else ""
            raise RuntimeError(
                f"No paired DICOM slices were found in '{self.root}' for "
                f"inputs {self.input_modalities}, target '{self.target_name}'{planct_msg}."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_paths, target_path, planct_path = self.pairs[idx]
        input_images = []
        for modality in self.input_modalities:
            modality_path = input_paths.get(modality)
            if modality == "CBCT" and self.use_planct_completion:
                input_images.append(self._load_completed_cbct_image(modality_path, planct_path))
            else:
                input_images.append(self._load_image(modality_path, modality))
        target_image = self._load_image(target_path, self.target_name)

        # Reuse the same RNG seed so paired medical slices receive identical
        # random spatial augmentation.
        seed = torch.randint(0, 2**32, (1,)).item()
        input_tensors = []
        for image in input_images:
            torch.manual_seed(seed)
            input_tensors.append(self.transform(image))
        item_a = torch.cat(input_tensors, dim=0)
        torch.manual_seed(seed)
        target = self.transform(target_image)
        return item_a, target

    def _build_transform(self, image_size, mode):
        transform = []
        if mode == "train":
            transform.append(T.RandomHorizontalFlip())
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
        return T.Compose(transform)

    def _find_pairs(self):
        pairs = []

        case_dirs = [path for path in self.root.iterdir() if path.is_dir()]
        has_flat_target = (self.root / self.target_name).is_dir()
        has_flat_planct = (self.root / self.planct_name).is_dir()
        has_flat_inputs = all((self.root / modality).is_dir() for modality in self.input_modalities)
        has_optional_cbct_completion = (
            self.use_planct_completion and has_flat_planct and all(
                modality == "CBCT" or (self.root / modality).is_dir()
                for modality in self.input_modalities
            )
        )
        if has_flat_target and (has_flat_inputs or has_optional_cbct_completion):
            case_dirs.append(self.root)

        for case_dir in sorted(set(case_dirs)):
            target_dir = case_dir / self.target_name
            planct_dir = case_dir / self.planct_name
            if not target_dir.is_dir():
                continue
            if self.use_planct_completion and not planct_dir.is_dir():
                continue

            input_files_by_modality = {}
            missing_required_input = False
            for modality in self.input_modalities:
                modality_dir = case_dir / modality
                if not modality_dir.is_dir():
                    if self.use_planct_completion and modality == "CBCT":
                        input_files_by_modality[modality] = {}
                        continue
                    missing_required_input = True
                    break
                input_files_by_modality[modality] = {path.name: path for path in sorted(modality_dir.glob("*.dcm"))}
            if missing_required_input:
                continue

            target_files = {path.name: path for path in sorted(target_dir.glob("*.dcm"))}
            common_names = set(target_files.keys())
            planct_files = {}
            if self.use_planct_completion:
                planct_files = {path.name: path for path in sorted(planct_dir.glob("*.dcm"))}
                common_names &= set(planct_files.keys())

            for modality, modality_files in input_files_by_modality.items():
                if self.use_planct_completion and modality == "CBCT":
                    continue
                common_names &= set(modality_files.keys())

            for name in sorted(common_names):
                input_paths = {}
                for modality, modality_files in input_files_by_modality.items():
                    input_paths[modality] = modality_files.get(name)
                pairs.append((input_paths, target_files[name], planct_files.get(name)))

        return pairs

    def _load_image(self, dicom_path, modality):
        pixels, raw_pixels = self._load_hu(dicom_path)
        del raw_pixels
        pixels = self._normalize_modality(pixels, modality)
        return self._to_grayscale_image(pixels)

    def _load_completed_cbct_image(self, cbct_path, planct_path):
        planct_hu, _ = self._load_hu(planct_path)
        planct_norm = self._normalize_modality(planct_hu, "CT")
        if cbct_path is None:
            return self._to_grayscale_image(planct_norm)

        cbct_hu, cbct_raw = self._load_hu(cbct_path)
        cbct_norm = self._normalize_modality(cbct_hu, "CBCT")
        fov_mask = self._build_fov_mask(cbct_hu, cbct_raw)
        completed = np.where(fov_mask, cbct_norm, planct_norm)
        return self._to_grayscale_image(completed)

    def _load_hu(self, dicom_path):
        dicom = pydicom.dcmread(str(dicom_path))
        raw_pixels = dicom.pixel_array.astype(np.float32)
        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        pixels = raw_pixels * slope + intercept
        return pixels, raw_pixels

    def _normalize_modality(self, pixels, modality):
        modality_key = modality.upper()
        if modality_key in CT_LIKE_MODALITIES:
            pixels = np.clip(pixels, -1000, 1000)
            return (pixels + 1000.0) / 2000.0

        pixels = np.clip(pixels, 0, 1500)
        return pixels / 1500.0

    def _build_fov_mask(self, cbct_hu, cbct_raw):
        if self.fov_mask_mode == "nonzero":
            return cbct_raw != 0
        if self.fov_mask_mode == "non_air":
            return cbct_hu > self.fov_threshold
        if self.fov_mask_mode == "all_cbct":
            return np.ones_like(cbct_hu, dtype=bool)
        raise ValueError(
            "fov_mask_mode must be one of: 'nonzero', 'non_air', 'all_cbct'"
        )

    def _to_grayscale_image(self, pixels):
        pixels = np.clip(pixels, 0.0, 1.0)
        return Image.fromarray((pixels * 255).astype(np.uint8), mode="L")


def get_loader(
    train_dir,
    itemA,
    image_size=256,
    batch_size=16,
    mode="train",
    num_workers=1,
    planct_name="PlanCT",
    use_planct_completion=False,
    fov_mask_mode="nonzero",
    fov_threshold=-950.0,
    input_modalities=None,
    target_name="CT",
):
    dataset = PairedDICOMFolder(
        train_dir,
        itemA,
        image_size=image_size,
        mode=mode,
        planct_name=planct_name,
        use_planct_completion=use_planct_completion,
        fov_mask_mode=fov_mask_mode,
        fov_threshold=fov_threshold,
        input_modalities=input_modalities,
        target_name=target_name,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
    )
