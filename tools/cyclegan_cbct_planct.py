#!/usr/bin/env python3
"""Convenience CLI for the CBCT + PlanCT CycleGAN workflow.

This wrapper keeps the common commands in one place:

1. prepare DICOM folders for train/validation/test
2. launch CBCT->sCT CycleGAN training with PlanCT completion
3. launch testing/inference from a saved checkpoint

It delegates the actual work to the existing repository scripts so model code and
preparation logic stay in their original modules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def str2bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "y"}


def run_command(command: List[str], dry_run: bool = False) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}")
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data" / "CBCT_CycleGAN")
    parser.add_argument("--runs_root", type=Path, default=PROJECT_ROOT / "runs" / "CycleCBCT_PlanCT")
    parser.add_argument("--planct_name", default="PlanCT")
    parser.add_argument("--input_modalities", default=None, help="comma-separated input folders, e.g. MRI,CBCT")
    parser.add_argument("--target_name", default="CT", help="target/supervision folder, e.g. CT or PlanCT")
    parser.add_argument("--use_planct_completion", type=str2bool, default=True)
    parser.add_argument("--fov_mask_mode", choices=["nonzero", "non_air", "all_cbct"], default="nonzero")
    parser.add_argument("--fov_threshold", type=float, default=-950.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="print command without executing it")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CBCT + PlanCT CycleGAN workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_split = subparsers.add_parser("prepare-split", help="prepare one split from raw case folders")
    prepare_split.add_argument("--input_root", type=Path, required=True)
    prepare_split.add_argument("--output_root", type=Path, required=True)
    prepare_split.add_argument("--key_mode", choices=["auto", "instance", "position"], default="auto")
    prepare_split.add_argument("--round_digits", type=int, default=3)
    prepare_split.add_argument("--dry_run", action="store_true", help="print command without executing it")

    prepare_all = subparsers.add_parser("prepare-all", help="prepare train/validation/test splits")
    prepare_all.add_argument("--raw_root", type=Path, required=True, help="root with train/, validation/, and test/proceeding/")
    prepare_all.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data" / "CBCT_CycleGAN")
    prepare_all.add_argument("--test_subdir", default="test/proceeding")
    prepare_all.add_argument("--key_mode", choices=["auto", "instance", "position"], default="auto")
    prepare_all.add_argument("--round_digits", type=int, default=3)
    prepare_all.add_argument("--dry_run", action="store_true", help="print commands without executing them")

    train = subparsers.add_parser("train", help="train CBCT->sCT CycleGAN with PlanCT completion")
    add_common_training_args(train)
    train.add_argument("--num_epochs", type=int, default=500)
    train.add_argument("--num_epochs_decay", type=int, default=200)
    train.add_argument("--g_lr", type=float, default=0.0001)
    train.add_argument("--d_lr", type=float, default=0.0001)
    train.add_argument("--use_tensorboard", type=str2bool, default=True)

    test = subparsers.add_parser("test", help="test/infer with a saved CBCT->sCT checkpoint")
    add_common_training_args(test)
    test.add_argument("--test_epochs", type=int, required=True, help="checkpoint epoch to load")
    test.add_argument("--use_tensorboard", type=str2bool, default=False)

    args = parser.parse_args()

    if args.command == "prepare-split":
        command = [
            sys.executable,
            "tools/prepare_cbct_planct_cyclegan.py",
            "--input_root", str(args.input_root),
            "--output_root", str(args.output_root),
            "--key_mode", args.key_mode,
            "--round_digits", str(args.round_digits),
        ]
        run_command(command, dry_run=args.dry_run)
        return

    if args.command == "prepare-all":
        split_pairs = [
            (args.raw_root / "train", args.data_root / "train"),
            (args.raw_root / "validation", args.data_root / "validation"),
            (args.raw_root / args.test_subdir, args.data_root / args.test_subdir),
        ]
        for input_root, output_root in split_pairs:
            command = [
                sys.executable,
                "tools/prepare_cbct_planct_cyclegan.py",
                "--input_root", str(input_root),
                "--output_root", str(output_root),
                "--key_mode", args.key_mode,
                "--round_digits", str(args.round_digits),
            ]
            run_command(command, dry_run=args.dry_run)
        return

    if args.command in {"train", "test"}:
        mode = args.command
        command = [
            sys.executable,
            "CycleGAN/main_CBCT.py",
            "--mode", mode,
            "--itemA", "CBCT",
            "--train_dir", str(args.data_root / "train"),
            "--val_dir", str(args.data_root / "validation"),
            "--test_dir", str(args.data_root / "test" / "proceeding"),
            "--log_dir", str(args.runs_root / "logs"),
            "--model_save_dir", str(args.runs_root / "models"),
            "--sample_dir", str(args.runs_root / "samples"),
            "--result_dir", str(args.runs_root / "results"),
            "--val_result_dir", str(args.runs_root / "val"),
            "--report_dir", str(args.runs_root / "report"),
            "--use_planct_completion", str(args.use_planct_completion).lower(),
            "--planct_name", args.planct_name,
            "--target_name", args.target_name,
            "--fov_mask_mode", args.fov_mask_mode,
            "--fov_threshold", str(args.fov_threshold),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--use_tensorboard", str(args.use_tensorboard).lower(),
        ]
        if args.input_modalities:
            command.extend(["--input_modalities", args.input_modalities])
        if mode == "train":
            command.extend([
                "--num_epochs", str(args.num_epochs),
                "--num_epochs_decay", str(args.num_epochs_decay),
                "--g_lr", str(args.g_lr),
                "--d_lr", str(args.d_lr),
            ])
        else:
            command.extend(["--test_epochs", str(args.test_epochs)])
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
