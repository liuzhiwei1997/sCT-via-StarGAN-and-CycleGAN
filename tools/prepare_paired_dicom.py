#!/usr/bin/env python3
"""Prepare paired MRI/CT DICOM slices for CycleGAN.

This script aligns two DICOM series by InstanceNumber (preferred) or
ImagePositionPatient(z) and writes matched pairs with identical filenames.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom


@dataclass
class SliceInfo:
    path: Path
    key: float


def read_series(series_dir: Path, key_mode: str) -> List[SliceInfo]:
    infos: List[SliceInfo] = []
    for path in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue

        key: Optional[float] = None
        if key_mode in {"instance", "auto"} and hasattr(ds, "InstanceNumber"):
            key = float(ds.InstanceNumber)

        if key is None and key_mode in {"position", "auto"} and hasattr(ds, "ImagePositionPatient"):
            pos = ds.ImagePositionPatient
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                key = float(pos[2])

        if key is not None:
            infos.append(SliceInfo(path=path, key=key))

    return infos


def build_map(infos: List[SliceInfo], round_digits: int) -> Dict[float, Path]:
    mapping: Dict[float, Path] = {}
    for info in infos:
        mapping[round(info.key, round_digits)] = info.path
    return mapping


def copy_pairs(
    mri_map: Dict[float, Path],
    ct_map: Dict[float, Path],
    output_case_dir: Path,
    mri_name: str,
    ct_name: str,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    common = sorted(set(mri_map.keys()) & set(ct_map.keys()))
    only_mri = len(set(mri_map.keys()) - set(ct_map.keys()))
    only_ct = len(set(ct_map.keys()) - set(mri_map.keys()))

    mri_out = output_case_dir / mri_name
    ct_out = output_case_dir / ct_name

    if not dry_run:
        mri_out.mkdir(parents=True, exist_ok=True)
        ct_out.mkdir(parents=True, exist_ok=True)

    for idx, key in enumerate(common, start=1):
        out_name = f"{idx:04d}.dcm"
        if not dry_run:
            shutil.copy2(mri_map[key], mri_out / out_name)
            shutil.copy2(ct_map[key], ct_out / out_name)

    return len(common), only_mri, only_ct


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired MRI/CT DICOM slices for CycleGAN")
    parser.add_argument("--mri_dir", required=True, type=Path, help="Input MRI series directory")
    parser.add_argument("--ct_dir", required=True, type=Path, help="Input CT series directory")
    parser.add_argument("--output_case_dir", required=True, type=Path, help="Output case dir, e.g. data/.../train/case001")
    parser.add_argument("--mri_name", default="MRI", help="Output MRI folder name")
    parser.add_argument("--ct_name", default="CT", help="Output CT folder name")
    parser.add_argument(
        "--key_mode",
        choices=["auto", "instance", "position"],
        default="auto",
        help="Slice match key: InstanceNumber, z-position, or auto",
    )
    parser.add_argument("--round_digits", type=int, default=3, help="Rounding for numeric key matching")
    parser.add_argument("--dry_run", action="store_true", help="Only print stats, do not copy files")

    args = parser.parse_args()

    if not args.mri_dir.is_dir() or not args.ct_dir.is_dir():
        raise SystemExit("--mri_dir and --ct_dir must be existing directories")

    mri_infos = read_series(args.mri_dir, args.key_mode)
    ct_infos = read_series(args.ct_dir, args.key_mode)

    mri_map = build_map(mri_infos, args.round_digits)
    ct_map = build_map(ct_infos, args.round_digits)

    paired, only_mri, only_ct = copy_pairs(
        mri_map,
        ct_map,
        args.output_case_dir,
        args.mri_name,
        args.ct_name,
        dry_run=args.dry_run,
    )

    print(f"MRI readable slices: {len(mri_map)}")
    print(f"CT readable slices:  {len(ct_map)}")
    print(f"Paired slices:       {paired}")
    print(f"MRI-only slices:     {only_mri}")
    print(f"CT-only slices:      {only_ct}")
    if not args.dry_run:
        print(f"Output: {args.output_case_dir / args.mri_name} and {args.output_case_dir / args.ct_name}")


if __name__ == "__main__":
    main()
