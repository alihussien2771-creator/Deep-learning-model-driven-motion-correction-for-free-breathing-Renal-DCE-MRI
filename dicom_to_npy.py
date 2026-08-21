"""
Convert raw MRI data to stack4d_before.npy files ready for training/inference.

Supported input formats:
  - DICOM        folder of .dcm files (or files with no extension)
  - NIfTI        .nii  /  .nii.gz
  - Philips PAR  .par  (paired with .rec)
  - NRRD         .nrrd
  - Analyze      .hdr  (paired with .img)

Usage via run.py (recommended — uses paths.py settings):
    Set MODE = "infer" or "test_infer" in paths.py. If DATA_ROOT has no
    converted scans yet, run.py converts RAW_INPUT automatically.

Direct CLI usage:
    # Single scan — DICOM folder
    python dicom_to_npy.py --input "D:/scans/patient_01/.../files" --output_dir "E:/data"

    # Multi-scan root folder (converts all scans found inside)
    python dicom_to_npy.py --input "D:/Exeter-patients1-20/input" --output_dir "E:/data"

For each scan, outputs are saved as:
    <output_dir>/<scan_name>/stack4d_before.npy
    <output_dir>/<scan_name>/crop_bounds.json
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Crop helper
# ─────────────────────────────────────────────────────────────────────────────

_CROP_THRESHOLD = 0.05   
_CROP_PAD       = 10     


def _auto_crop(volume: np.ndarray):
    mip  = np.max(volume, axis=(0, 1))          
    mask = mip > (_CROP_THRESHOLD * mip.max())
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        print("[Crop] No foreground found — skipping crop.")
        return volume, (0, mip.shape[0], 0, mip.shape[1])
    rmin = max(0,            np.where(rows)[0][0]  - _CROP_PAD)
    rmax = min(mip.shape[0], np.where(rows)[0][-1] + 1 + _CROP_PAD)
    cmin = max(0,            np.where(cols)[0][0]  - _CROP_PAD)
    cmax = min(mip.shape[1], np.where(cols)[0][-1] + 1 + _CROP_PAD)
    return volume[:, :, rmin:rmax, cmin:cmax], (rmin, rmax, cmin, cmax)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scan discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_dicom_folders(root: str) -> list:
    
    def _has_readable_dicom_files(dirpath: str, filenames: list[str]) -> bool:
        if not filenames:
            return False

        # Check likely DICOM files first, then extensionless files, then a few
        # remaining files. Many XNAT exports use no .dcm extension.
        def priority(name: str) -> tuple[int, str]:
            suffix = Path(name).suffix.lower()
            if suffix == ".dcm":
                return (0, name)
            if suffix == "":
                return (1, name)
            return (2, name)

        candidates = sorted(filenames, key=priority)
        candidates = [n for n in candidates if not n.startswith(".")][:25]

        try:
            import pydicom
        except ImportError:
            return any(Path(n).suffix.lower() == ".dcm" for n in candidates)

        for name in candidates:
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):
                continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
                    return True
            except Exception:
                continue
        return False

    xnat_found = []
    direct_found = []
    root_norm = os.path.normpath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune SNAPSHOT branches so os.walk never descends into them
        dirnames[:] = [d for d in dirnames if "snapshot" not in d.lower()]

        parts = Path(os.path.normpath(dirpath)).parts
        if (len(parts) >= 3
                and parts[-1].lower() == "files"
                and parts[-2].lower() == "dicom"
                and parts[-3].lower() == "resources"
                and filenames):   # must actually contain files
            xnat_found.append(dirpath)
            continue

        if _has_readable_dicom_files(dirpath, filenames):
            direct_found.append(os.path.normpath(dirpath))

    # Prefer the explicit XNAT folders when present.
    if xnat_found:
        return sorted(set(xnat_found))

    
    direct_found = sorted(set(direct_found))
    leaf_folders = []
    for candidate in direct_found:
        prefix = candidate + os.sep
        has_child_candidate = any(
            other != candidate and other.startswith(prefix)
            for other in direct_found
        )
        if not has_child_candidate:
            leaf_folders.append(candidate)

    if len(leaf_folders) > 1:
        leaf_folders = [p for p in leaf_folders if p != root_norm]

    return sorted(leaf_folders)


def _patient_name_from_path(dicom_folder: str, raw_root: str | None = None) -> str:
    
    norm  = os.path.normpath(dicom_folder)
    path = Path(norm)

    # Direct file input: use the filename without all suffixes
    # (scan.nii.gz -> scan, scan.par -> scan).
    if path.is_file():
        name = path.name
        for suffix in path.suffixes:
            name = name[: -len(suffix)]
        if name:
            return name

    parts = Path(norm).parts

    # Strategy 1: first component relative to the root
    if raw_root:
        try:
            rel   = os.path.relpath(norm, os.path.normpath(raw_root))
            first = Path(rel).parts[0]
            if first not in (".", "..") and first:
                return first
        except ValueError:
            pass   

    # Strategy 2: parent of the "scans" directory (typical MDR layout)
    for i, part in enumerate(parts):
        if part.lower() == "scans" and i > 0:
            return parts[i - 1]

    # Strategy 3: deepest component that is not a known DICOM path keyword
    skip = {"files", "dicom", "resources"}
    for part in reversed(parts):
        if part.lower() not in skip and part:
            return part

    return "scan_001"


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion function
# ─────────────────────────────────────────────────────────────────────────────

def convert_one_scan(input_path: str, output_dir: str) -> bool:
    
    from mri_reader import read_any_mri

    try:
        os.makedirs(output_dir, exist_ok=True)

        volume = read_any_mri(input_path)

       
        T, Z, H, W = volume.shape
        bounds = (0, H, 0, W)
        rmin, rmax, cmin, cmax = bounds
        print("[Crop] Auto-crop disabled - keeping full image.")
        print(f"[Crop] Bounding box : rows {rmin}:{rmax},  cols {cmin}:{cmax}")
        print(f"[Crop] Full shape: {volume.shape}")

        out_npy    = os.path.join(output_dir, "stack4d_before.npy")
        out_bounds = os.path.join(output_dir, "crop_bounds.json")

        np.save(out_npy, volume)
        with open(out_bounds, "w", encoding="utf-8") as f:
            json.dump({"rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax}, f)

        print(f"  Saved : {out_npy}")
        print(f"  Shape : (T={T}, Z={Z}, H={H}, W={W})")
        print(f"  Range : [{volume.min():.1f}, {volume.max():.1f}]")
        return True

    except Exception as exc:
        print(f"  [FAILED] {type(exc).__name__}: {exc}")
        return False




def convert_batch(raw_input: str, data_root: str) -> None:
   
    raw_input = os.path.normpath(raw_input)
    data_root = os.path.normpath(data_root)

    if not (os.path.isdir(raw_input) or os.path.isfile(raw_input)):
        raise FileNotFoundError(
            f"RAW_INPUT does not exist: {raw_input}\n"
            "Check the RAW_INPUT setting in paths.py."
        )

    # Detect single vs. batch
    is_file_input = os.path.isfile(raw_input)
    raw_parts     = Path(raw_input).parts
    is_files_dir  = (len(raw_parts) >= 3
                     and raw_parts[-1].lower() == "files"
                     and raw_parts[-2].lower() == "dicom"
                     and raw_parts[-3].lower() == "resources")

    dicom_folders = [] if (is_file_input or is_files_dir) else _find_dicom_folders(raw_input)

    if is_file_input or is_files_dir or not dicom_folders:
        # ── Single-scan mode ──────────────────────────────────────────────
        src        = raw_input
        scan_name  = _patient_name_from_path(src, None)
        output_dir = os.path.join(data_root, scan_name)

        print()
        print("Single-scan mode")
        print(f"  Input  : {src}")
        print(f"  Output : {output_dir}")
        if is_file_input:
            print("  (Treating RAW_INPUT as a single MRI file)")
        elif not is_files_dir and not dicom_folders:
            print("  (No resources/DICOM/files sub-folders found — "
                  "treating RAW_INPUT as a DICOM folder directly)")
        print()

        out_npy = os.path.join(output_dir, "stack4d_before.npy")
        if os.path.isfile(out_npy):
            print(f"  [SKIP]  stack4d_before.npy already exists: {out_npy}")
            converted, skipped, failed = 0, 1, 0
        else:
            ok        = convert_one_scan(src, output_dir)
            converted = 1 if ok else 0
            skipped   = 0
            failed    = 0 if ok else 1

    else:
        # ── Batch mode ────────────────────────────────────────────────────
        print()
        print(f"Batch mode — found {len(dicom_folders)} scan folder(s)")
        print(f"  Source : {raw_input}")
        print(f"  Output : {data_root}")
        print()

        # Pre-compute all scan names and deduplicate
        raw_names  = [_patient_name_from_path(f, raw_input) for f in dicom_folders]
        name_count = Counter(raw_names)
        name_seen  = Counter()

        scan_jobs = []    # list of (dicom_folder, scan_name, output_dir)
        for dicom_folder, raw_name in zip(dicom_folders, raw_names):
            name_seen[raw_name] += 1
            if name_count[raw_name] > 1:
                scan_name = f"{raw_name}_{name_seen[raw_name]:02d}"
            else:
                scan_name = raw_name
            scan_jobs.append((dicom_folder, scan_name,
                               os.path.join(data_root, scan_name)))

        converted, skipped, failed = 0, 0, 0
        for i, (dicom_folder, scan_name, output_dir) in enumerate(scan_jobs, start=1):
            print(f"[{i}/{len(scan_jobs)}]  {scan_name}")
            print(f"  Input  : {dicom_folder}")
            print(f"  Output : {output_dir}")

            out_npy = os.path.join(output_dir, "stack4d_before.npy")
            if os.path.isfile(out_npy):
                print(f"  [SKIP]  stack4d_before.npy already exists.")
                skipped += 1
                print()
                continue

            ok = convert_one_scan(dicom_folder, output_dir)
            if ok:
                converted += 1
            else:
                failed += 1
            print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Conversion complete")
    print(f"  Converted : {converted}")
    print(f"  Skipped   : {skipped}  (stack4d_before.npy already existed)")
    print(f"  Failed    : {failed}")
    print(f"  Output    : {data_root}")
    print("=" * 60)




def parse_args() -> argparse.Namespace:
    import paths as _paths
    p = argparse.ArgumentParser(
        description="Convert raw MRI data to stack4d_before.npy"
    )
    p.add_argument(
        "--input",
        default=_paths.RAW_INPUT,
        help="DICOM folder, NIfTI file, PAR file, or a root folder of scans",
    )
    p.add_argument(
        "--output_dir",
        default=_paths.DATA_ROOT,
        help="Destination root folder — each scan gets its own sub-folder",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convert_batch(args.input, args.output_dir)


if __name__ == "__main__":
    main()
