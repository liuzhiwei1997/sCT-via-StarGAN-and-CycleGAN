#!/usr/bin/env python3
"""Register and resample case DICOM series to a common reference grid.

Typical use before MRI+CBCT->sCT training with PlanCT as target:

    raw/Case001/{MRI,CBCT,PlanCT} -> registered/Case001/{MRI,CBCT,PlanCT}

`PlanCT` is the default fixed/reference series. Moving modalities are registered
with 3D mutual-information rigid registration and then resampled onto the fixed
voxel grid. The output series are synchronized by fixed-grid slice index so the
CycleGAN loaders can pair slices by filename.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def parse_modalities(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_dicom_series(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not names:
        raise RuntimeError(f"No DICOM series found in {series_dir}")
    reader.SetFileNames(names)
    return reader.Execute()


def as_float(image: sitk.Image) -> sitk.Image:
    return sitk.Cast(image, sitk.sitkFloat32)


def make_transform(fixed: sitk.Image, moving: sitk.Image, mode: str) -> sitk.Transform:
    if mode == "none":
        return sitk.Transform(3, sitk.sitkIdentity)

    transform_type = sitk.Euler3DTransform() if mode == "rigid" else sitk.AffineTransform(3)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform_type,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    return registration.Execute(as_float(fixed), as_float(moving))


def get_interpolator(name: str) -> int:
    if name == "nearest":
        return sitk.sitkNearestNeighbor
    if name == "bspline":
        return sitk.sitkBSpline
    return sitk.sitkLinear


def resample_to_fixed(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
    interpolator: int,
    default_value: float,
) -> sitk.Image:
    return sitk.Resample(
        moving,
        fixed,
        transform,
        interpolator,
        default_value,
        moving.GetPixelID(),
    )


def first_dicom_metadata(series_dir: Path) -> pydicom.Dataset | None:
    for path in sorted(series_dir.glob("*.dcm")):
        try:
            return pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
    return None


def image_position(image: sitk.Image, slice_index: int) -> str:
    point = image.TransformIndexToPhysicalPoint((0, 0, slice_index))
    return "\\".join(f"{coord:.6f}" for coord in point)


def image_orientation(image: sitk.Image) -> str:
    direction = image.GetDirection()
    row = (direction[0], direction[3], direction[6])
    col = (direction[1], direction[4], direction[7])
    return "\\".join(f"{coord:.6f}" for coord in (*row, *col))


def write_dicom_series(image: sitk.Image, output_dir: Path, reference_dir: Path, modality_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = first_dicom_metadata(reference_dir)
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array = np.nan_to_num(array)
    array = np.clip(np.rint(array), -32768, 32767).astype(np.int16)

    spacing = image.GetSpacing()
    series_uid = generate_uid()
    study_uid = getattr(reference, "StudyInstanceUID", generate_uid()) if reference is not None else generate_uid()
    frame_uid = getattr(reference, "FrameOfReferenceUID", generate_uid()) if reference is not None else generate_uid()
    patient_name = getattr(reference, "PatientName", "Anonymous") if reference is not None else "Anonymous"
    patient_id = getattr(reference, "PatientID", "Unknown") if reference is not None else "Unknown"

    for z_index, slice_array in enumerate(array):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(output_dir / f"{z_index + 1:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = getattr(reference, "Modality", "OT") if reference is not None else "OT"
        ds.SeriesDescription = f"Registered_{modality_name}"
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.Rows, ds.Columns = slice_array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.InstanceNumber = z_index + 1
        ds.ImagePositionPatient = image_position(image, z_index)
        ds.ImageOrientationPatient = image_orientation(image)
        ds.PixelSpacing = [str(spacing[1]), str(spacing[0])]
        ds.SliceThickness = str(spacing[2])
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0
        ds.PixelData = slice_array.tobytes()
        ds.save_as(output_dir / f"{z_index + 1:04d}.dcm")


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_or_write_fixed(fixed_dir: Path, output_dir: Path, clean_output: bool) -> None:
    if clean_output:
        clean_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(sorted(fixed_dir.glob("*.dcm")), start=1):
        shutil.copy2(path, output_dir / f"{idx:04d}.dcm")


def process_case(case_dir: Path, output_case_dir: Path, args: argparse.Namespace) -> None:
    fixed_dir = case_dir / args.fixed_name
    fixed = read_dicom_series(fixed_dir)
    fixed_output = output_case_dir / args.fixed_name
    copy_or_write_fixed(fixed_dir, fixed_output, clean_output=not args.dry_run)
    print(f"[{case_dir.name}] Fixed {args.fixed_name}: {fixed.GetSize()} -> {fixed_output}")

    for modality in args.moving_modalities:
        moving_dir = case_dir / modality
        if not moving_dir.is_dir():
            print(f"[{case_dir.name}] Skip missing moving modality: {modality}")
            continue
        output_dir = output_case_dir / modality
        if args.dry_run:
            print(f"[{case_dir.name}] Would register {modality} -> {args.fixed_name}: {output_dir}")
            continue
        clean_dir(output_dir)
        moving = read_dicom_series(moving_dir)
        transform = make_transform(fixed, moving, args.transform)
        resampled = resample_to_fixed(
            moving,
            fixed,
            transform,
            get_interpolator(args.interpolator),
            args.default_value,
        )
        write_dicom_series(resampled, output_dir, moving_dir, modality)
        print(f"[{case_dir.name}] Registered {modality}: {moving.GetSize()} -> {resampled.GetSize()} at {output_dir}")


def iter_case_dirs(input_root: Path) -> Iterable[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Register/resample DICOM case folders to a fixed reference grid")
    parser.add_argument("--input_root", type=Path, required=True, help="root with case folders")
    parser.add_argument("--output_root", type=Path, required=True, help="output root for registered case folders")
    parser.add_argument("--fixed_name", default="PlanCT", help="reference/fixed folder name")
    parser.add_argument("--moving_modalities", default="MRI,CBCT", help="comma-separated moving folders to register")
    parser.add_argument("--transform", choices=["rigid", "affine", "none"], default="rigid")
    parser.add_argument("--interpolator", choices=["linear", "nearest", "bspline"], default="linear")
    parser.add_argument("--default_value", type=float, default=-1000.0, help="fill value outside moving image support")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.moving_modalities = parse_modalities(args.moving_modalities)

    if not args.input_root.is_dir():
        raise SystemExit("--input_root must be an existing directory")
    args.output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for case_dir in iter_case_dirs(args.input_root):
        if not (case_dir / args.fixed_name).is_dir():
            skipped += 1
            print(f"[{case_dir.name}] Skipped: missing fixed folder {args.fixed_name}")
            continue
        process_case(case_dir, args.output_root / case_dir.name, args)
        processed += 1
    print(f"Registration summary: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
