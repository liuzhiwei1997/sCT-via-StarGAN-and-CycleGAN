"""Paired DICOM loader used by the CycleGAN training scripts."""

from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class PairedDICOMFolder(data.Dataset):
    """Load paired modality slices for CycleGAN training.

    Expected layouts:

    * ``root/CASE_ID/<itemA>/*.dcm`` and ``root/CASE_ID/CT/*.dcm``
    * ``root/<itemA>/*.dcm`` and ``root/CT/*.dcm``

    Pairing is performed by filename within each case directory.
    """

    def __init__(self, root, item_a, image_size=(512, 512), mode="train"):
        self.root = Path(root)
        self.item_a = item_a
        self.mode = mode
        self.transform = self._build_transform(image_size, mode)
        self.pairs = self._find_pairs()

        if not self.pairs:
            raise RuntimeError(
                f"No paired DICOM slices were found in '{self.root}' for "
                f"modalities '{self.item_a}' and 'CT'."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item_a_path, ct_path = self.pairs[idx]
        item_a = self.transform(self._load_image(item_a_path, self.item_a))
        ct = self.transform(self._load_image(ct_path, "CT"))
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
            if not item_dir.is_dir() or not ct_dir.is_dir():
                continue

            item_files = {path.name: path for path in sorted(item_dir.glob("*.dcm"))}
            ct_files = {path.name: path for path in sorted(ct_dir.glob("*.dcm"))}
            for name in sorted(item_files.keys() & ct_files.keys()):
                pairs.append((item_files[name], ct_files[name]))

        return pairs

    def _load_image(self, dicom_path, modality):
        dicom = pydicom.dcmread(str(dicom_path))
        pixels = dicom.pixel_array.astype(np.float32)
        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept

        if modality == "CT":
            pixels = np.clip(pixels, -1000, 1000)
            pixels = (pixels + 1000.0) / 2000.0
        else:
            pixels = np.clip(pixels, 0, 1500)
            pixels = pixels / 1500.0

        image = Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
        return image


def get_loader(train_dir, itemA, image_size=256, batch_size=16, mode="train", num_workers=1):
    dataset = PairedDICOMFolder(train_dir, itemA, image_size=image_size, mode=mode)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
    )
