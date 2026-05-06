#!/usr/bin/env python3
"""Prepare CBCT/PlanCT/CT DICOM triplets for CycleGAN.

The CycleGAN CBCT pipeline expects identical filenames under each case:

    case001/CBCT/0001.dcm
    case001/PlanCT/0001.dcm
    case001/CT/0001.dcm

This script aligns the three input series by InstanceNumber or z-position,
keeps only the common slices, and copies them into the simplified layout above.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def build_map(infos: Iterable[SliceInfo], round_digits: int) -> Dict[float, Path]:
    mapping: Dict[float, Path] = {}
    for info in infos:
        mapping[round(info.key, round_digits)] = info.path
    return mapping


def copy_triplets(
    cbct_map: Dict[float, Path],
    planct_map: Dict[float, Path],
    ct_map: Dict[float, Path],
    output_case_dir: Path,
    cbct_name: str,
    planct_name: str,
    ct_name: str,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    cbct_keys = set(cbct_map.keys())
    planct_keys = set(planct_map.keys())
    ct_keys = set(ct_map.keys())
    common = sorted(cbct_keys & planct_keys & ct_keys)

    cbct_only = len(cbct_keys - planct_keys - ct_keys)
    planct_only = len(planct_keys - cbct_keys - ct_keys)
    ct_only = len(ct_keys - cbct_keys - planct_keys)

    cbct_out = output_case_dir / cbct_name
    planct_out = output_case_dir / planct_name
    ct_out = output_case_dir / ct_name

    if not dry_run:
        cbct_out.mkdir(parents=True, exist_ok=True)
        planct_out.mkdir(parents=True, exist_ok=True)
        ct_out.mkdir(parents=True, exist_ok=True)

    for idx, key in enumerate(common, start=1):
        out_name = f"{idx:04d}.dcm"
        if not dry_run:
            shutil.copy2(cbct_map[key], cbct_out / out_name)
            shutil.copy2(planct_map[key], planct_out / out_name)
            shutil.copy2(ct_map[key], ct_out / out_name)

    return len(common), cbct_only, planct_only, ct_only


def process_one_case(
    cbct_dir: Path,
    planct_dir: Path,
    ct_dir: Path,
    output_case_dir: Path,
    case_name: str,
    args: argparse.Namespace,
) -> None:
    cbct_map = build_map(read_series(cbct_dir, args.key_mode), args.round_digits)
    planct_map = build_map(read_series(planct_dir, args.key_mode), args.round_digits)
    ct_map = build_map(read_series(ct_dir, args.key_mode), args.round_digits)

    triplets, cbct_only, planct_only, ct_only = copy_triplets(
        cbct_map,
        planct_map,
        ct_map,
        output_case_dir,
        args.cbct_name,
        args.planct_name,
        args.ct_name,
        dry_run=args.dry_run,
    )

    print(f"[{case_name}] CBCT readable slices:   {len(cbct_map)}")
    print(f"[{case_name}] PlanCT readable slices: {len(planct_map)}")
    print(f"[{case_name}] CT readable slices:     {len(ct_map)}")
    print(f"[{case_name}] Matched triplets:       {triplets}")
    print(f"[{case_name}] CBCT-only slices:       {cbct_only}")
    print(f"[{case_name}] PlanCT-only slices:     {planct_only}")
    print(f"[{case_name}] CT-only slices:         {ct_only}")
    if not args.dry_run:
        print(f"[{case_name}] Output case: {output_case_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CBCT/PlanCT/CT DICOM triplets for CycleGAN CBCT-to-sCT training"
    )
    parser.add_argument("--cbct_dir", type=Path, help="Single-case input CBCT series directory")
    parser.add_argument("--planct_dir", type=Path, help="Single-case input PlanCT series directory")
    parser.add_argument("--ct_dir", type=Path, help="Single-case input target CT series directory")
    parser.add_argument("--output_case_dir", type=Path, help="Single-case output dir, e.g. data/.../train/case001")
    parser.add_argument("--input_root", type=Path, help="Batch root with case folders")
    parser.add_argument("--output_root", type=Path, help="Batch output root where case folders are created")
    parser.add_argument("--cbct_name", default="CBCT", help="CBCT folder name in input and output cases")
    parser.add_argument("--planct_name", default="PlanCT", help="PlanCT folder name in input and output cases")
    parser.add_argument("--ct_name", default="CT", help="Target CT folder name in input and output cases")
    parser.add_argument(
        "--key_mode",
        choices=["auto", "instance", "position"],
        default="auto",
        help="Slice match key: InstanceNumber, z-position, or auto",
    )
    parser.add_argument("--round_digits", type=int, default=3, help="Rounding for numeric key matching")
    parser.add_argument("--dry_run", action="store_true", help="Print stats without copying files")
    args = parser.parse_args()

    single_mode = any([args.cbct_dir, args.planct_dir, args.ct_dir, args.output_case_dir])
    batch_mode = any([args.input_root, args.output_root])
    if single_mode and batch_mode:
        raise SystemExit("Use either single-case args or batch args, not both")

    if batch_mode:
        if args.input_root is None or args.output_root is None:
            raise SystemExit("Batch mode requires both --input_root and --output_root")
        if not args.input_root.is_dir():
            raise SystemExit("--input_root must be an existing directory")

        processed = 0
        skipped = 0
        for case_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
            cbct_dir = case_dir / args.cbct_name
            planct_dir = case_dir / args.planct_name
            ct_dir = case_dir / args.ct_name
            if not cbct_dir.is_dir() or not planct_dir.is_dir() or not ct_dir.is_dir():
                skipped += 1
                print(
                    f"[{case_dir.name}] Skipped: expected '{args.cbct_name}', "
                    f"'{args.planct_name}', and '{args.ct_name}' folders"
                )
                continue
            process_one_case(cbct_dir, planct_dir, ct_dir, args.output_root / case_dir.name, case_dir.name, args)
            processed += 1

        print(f"Batch summary: processed={processed}, skipped={skipped}")
        return

    if args.cbct_dir is None or args.planct_dir is None or args.ct_dir is None or args.output_case_dir is None:
        raise SystemExit("Single-case mode requires --cbct_dir, --planct_dir, --ct_dir, and --output_case_dir")
    if not args.cbct_dir.is_dir() or not args.planct_dir.is_dir() or not args.ct_dir.is_dir():
        raise SystemExit("--cbct_dir, --planct_dir, and --ct_dir must be existing directories")
    process_one_case(args.cbct_dir, args.planct_dir, args.ct_dir, args.output_case_dir, "single_case", args)


if __name__ == "__main__":
    main()
