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
    ``PlanCT`` folder with filenames matching ``itemA`` and ``CT``. The returned
    input image is ``CBCT`` inside the detected FOV and ``PlanCT`` outside it.
    Pairing is performed by filename within each case directory.
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
    ):
        self.root = Path(root)
        self.item_a = item_a
        self.mode = mode
        self.planct_name = planct_name
        self.use_planct_completion = use_planct_completion
        self.fov_mask_mode = fov_mask_mode
        self.fov_threshold = fov_threshold
        self.transform = self._build_transform(image_size, mode)
        self.pairs = self._find_pairs()

        if not self.pairs:
            planct_msg = f", '{self.planct_name}'" if self.use_planct_completion else ""
            raise RuntimeError(
                f"No paired DICOM slices were found in '{self.root}' for "
                f"modalities '{self.item_a}', 'CT'{planct_msg}."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item_a_path, ct_path, planct_path = self.pairs[idx]
        if self.use_planct_completion:
            item_a_image = self._load_completed_cbct_image(item_a_path, planct_path)
        else:
            item_a_image = self._load_image(item_a_path, self.item_a)
        ct_image = self._load_image(ct_path, "CT")

        # Reuse the same RNG seed so paired medical slices receive identical
        # random spatial augmentation.
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        item_a = self.transform(item_a_image)
        torch.manual_seed(seed)
        ct = self.transform(ct_image)
        return item_a, ct

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
        if (self.root / self.item_a).is_dir() and (self.root / "CT").is_dir():
            case_dirs.append(self.root)

        for case_dir in sorted(set(case_dirs)):
            item_dir = case_dir / self.item_a
            ct_dir = case_dir / "CT"
            planct_dir = case_dir / self.planct_name
            if not item_dir.is_dir() or not ct_dir.is_dir():
                continue
            if self.use_planct_completion and not planct_dir.is_dir():
                continue

            item_files = {path.name: path for path in sorted(item_dir.glob("*.dcm"))}
            ct_files = {path.name: path for path in sorted(ct_dir.glob("*.dcm"))}
            common_names = item_files.keys() & ct_files.keys()

            planct_files = {}
            if self.use_planct_completion:
                planct_files = {path.name: path for path in sorted(planct_dir.glob("*.dcm"))}
                common_names = common_names & planct_files.keys()

            for name in sorted(common_names):
                pairs.append((item_files[name], ct_files[name], planct_files.get(name)))

        return pairs

    def _load_image(self, dicom_path, modality):
        pixels, raw_pixels = self._load_hu(dicom_path)
        del raw_pixels
        pixels = self._normalize_modality(pixels, modality)
        return self._to_grayscale_image(pixels)

    def _load_completed_cbct_image(self, cbct_path, planct_path):
        cbct_hu, cbct_raw = self._load_hu(cbct_path)
        planct_hu, _ = self._load_hu(planct_path)
        cbct_norm = self._normalize_modality(cbct_hu, "CBCT")
        planct_norm = self._normalize_modality(planct_hu, "CT")
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
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
    )
