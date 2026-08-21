"""
Unified MRI reader for the renal DCE-MRI DL pipeline.

Supports:
  - DICOM          folder of .dcm files (or files with no extension)
  - NIfTI          .nii  /  .nii.gz
  - Philips PAR    .par  (paired with .rec)
  - NRRD           .nrrd
  - Analyze        .hdr  (paired with .img)

Always returns a float32 numpy array with shape (T, Z, H, W).
For static 3D volumes (no time axis) T is set to 1.
"""

from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


# ── Format detection ─────────────────────────────────────────────────────────

def _detect_format(input_path: str) -> str:
    """Return one of: 'dicom', 'nifti', 'parrec', 'nrrd', 'analyze'."""
    p = Path(input_path)

    # File supplied directly
    if p.is_file():
        ext = "".join(p.suffixes).lower()   # handles .nii.gz
        if ext in (".nii", ".nii.gz"):
            return "nifti"
        if ext == ".par":
            return "parrec"
        if ext == ".nrrd":
            return "nrrd"
        if ext == ".hdr":
            return "analyze"
        if ext == ".dcm":
            return "dicom"
        # Unknown extension — try nibabel, then DICOM
        return "unknown_file"

    # Folder supplied — look at file extensions inside
    if p.is_dir():
        exts = {Path(f).suffix.lower()
                for f in os.listdir(p)
                if os.path.isfile(os.path.join(p, f))}

        if ".nii" in exts or ".gz" in exts:
            return "nifti"
        if ".par" in exts:
            return "parrec"
        if ".nrrd" in exts:
            return "nrrd"
        if ".hdr" in exts:
            return "analyze"
        # Default: treat as DICOM folder
        return "dicom"

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


# ── DICOM helpers (ported from MDR pipeline) ──────────────────────────────────

def _get_series_uid(ds) -> str:
    return str(ds.get("SeriesInstanceUID", ""))

def _get_iop(ds):
    iop = ds.get("ImageOrientationPatient", None)
    return tuple(round(float(v), 6) for v in iop) if iop else None

def _get_z(ds) -> float:
    ipp = ds.get("ImagePositionPatient", None)
    return float(ipp[2]) if (ipp and len(ipp) >= 3) else 0.0


# ── DICOM reader ──────────────────────────────────────────────────────────────

def _read_dicom(input_path: str) -> np.ndarray:
    import pydicom

    p = Path(input_path)
    folder = p if p.is_dir() else p.parent

    # ── Collect and filter DICOM headers ────────────────────────────────────
    all_files = [str(fp) for fp in folder.rglob("*") if fp.is_file()]
    print(f"[DICOM] Found {len(all_files)} files in {folder}")

    dicoms = []
    for i, path in enumerate(all_files):
        if i > 0 and i % 500 == 0:
            print(f"  Reading headers {i}/{len(all_files)} ...")
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
                ds._filepath = path
                dicoms.append(ds)
        except Exception:
            continue

    if not dicoms:
        raise RuntimeError(f"No readable DICOM files found in {folder}")

    # Filter 1: dominant series (removes secondary captures, reports, etc.)
    series_counts = Counter(_get_series_uid(d) for d in dicoms)
    main_series = series_counts.most_common(1)[0][0]
    dicoms = [d for d in dicoms if _get_series_uid(d) == main_series]

    # Filter 2: dominant image orientation (removes localisers/scouts)
    iop_counts = Counter(_get_iop(d) for d in dicoms)
    main_iop = iop_counts.most_common(1)[0][0]
    dicoms = [d for d in dicoms if _get_iop(d) == main_iop]

    print(f"[DICOM] After filtering: {len(dicoms)} images kept")
    N = len(dicoms)

    # ── Build per-image records ──────────────────────────────────────────────
    records = []
    for ds in dicoms:
        # Slice position (Z)
        z_pos = getattr(ds, "SliceLocation", None)
        if z_pos is None:
            z_pos = _get_z(ds)

        # Temporal metadata — only the reliable dedicated tags
        t_meta = None
        for tag in ("TemporalPositionIdentifier", "TemporalPositionIndex"):
            val = getattr(ds, tag, None)
            if val is not None:
                t_meta = float(val)
                break

        # Acquisition order — InstanceNumber is most reliable for ordering
        instance = int(getattr(ds, "InstanceNumber", 0))

        records.append({
            "path":     ds._filepath,
            "z_pos":    float(z_pos) if z_pos is not None else 0.0,
            "t_meta":   t_meta,           # None if tag absent
            "instance": instance,
        })

    unique_z  = sorted(set(r["z_pos"]  for r in records))
    Z         = len(unique_z)
    z_map     = {v: i for i, v in enumerate(unique_z)}

    

    meta_values = [r["t_meta"] for r in records if r["t_meta"] is not None]
    n_unique_meta = len(set(meta_values))

    use_order = True      # default: safer
    if len(meta_values) == N and n_unique_meta < N and (N % Z == 0):
        # All images have the tag, fewer unique values than images,
        # and the maths works out — metadata looks reliable.
        T_check = n_unique_meta
        if T_check * Z == N:
            use_order = False

    if use_order:
        if N % Z != 0:
            raise RuntimeError(
                f"[DICOM] Cannot group {N} images into slices of Z={Z} — "
                f"{N} is not divisible by {Z}.\n"
                f"Check that the folder contains exactly one DCE series."
            )
        T = N // Z
        grouping_method = "order-based grouping (acquisition order by InstanceNumber)"
    else:
        T = n_unique_meta
        grouping_method = "metadata time grouping (TemporalPositionIdentifier/Index)"

    # ── Read a sample image to get H, W ─────────────────────────────────────
    sample = pydicom.dcmread(records[0]["path"])
    H, W   = sample.pixel_array.shape

    print(f"[DICOM] Grouping : {grouping_method}")
    print(f"[DICOM] T={T}  Z={Z}  H={H}  W={W}")

    if T > 2000:
        print(f"[DICOM] WARNING: T={T} is unexpectedly large.")
        print(f"[DICOM]   This may mean grouping failed — check the input folder.")

    # ── Read all pixel data into the volume ──────────────────────────────────
    volume  = np.zeros((T, Z, H, W), dtype=np.float32)

    if use_order:
        # Sort all records by InstanceNumber, then chunk into T groups of Z.
        # Within each chunk, sort by z_pos to get a consistent slice ordering.
        records_sorted = sorted(records, key=lambda r: r["instance"])
        flat = 0
        for t_idx in range(T):
            chunk = records_sorted[t_idx * Z : (t_idx + 1) * Z]
            chunk = sorted(chunk, key=lambda r: r["z_pos"])
            for z_idx, rec in enumerate(chunk):
                if flat % 200 == 0:
                    print(f"  Reading pixels {flat}/{N} ...")
                flat += 1
                ds  = pydicom.dcmread(rec["path"])
                pix = ds.pixel_array.astype(np.float32)
                pix = pix * float(getattr(ds, "RescaleSlope", 1.0)) \
                          + float(getattr(ds, "RescaleIntercept", 0.0))
                volume[t_idx, z_idx] = pix
    else:
        # Use temporal metadata tags directly.
        meta_unique = sorted(set(r["t_meta"] for r in records))
        t_map = {v: i for i, v in enumerate(meta_unique)}
        for i, rec in enumerate(records):
            if i % 200 == 0:
                print(f"  Reading pixels {i}/{N} ...")
            ds  = pydicom.dcmread(rec["path"])
            pix = ds.pixel_array.astype(np.float32)
            pix = pix * float(getattr(ds, "RescaleSlope", 1.0)) \
                      + float(getattr(ds, "RescaleIntercept", 0.0))
            volume[t_map[rec["t_meta"]], z_map[rec["z_pos"]]] = pix

    print(f"[DICOM] Volume shape: {volume.shape}")
    return volume


# ── NIfTI / PAR-REC / NRRD / Analyze reader ──────────────────────────────────

def _read_nibabel(input_path: str, fmt: str) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI/PAR-REC/NRRD/Analyze files.\n"
            "Install it with:  pip install nibabel"
        )

    p = Path(input_path)

    # If a folder was given, find the relevant file inside it
    if p.is_dir():
        ext_map = {
            "nifti":   [".nii", ".gz"],
            "parrec":  [".par"],
            "nrrd":    [".nrrd"],
            "analyze": [".hdr"],
        }
        targets = ext_map.get(fmt, [])
        candidates = [
            p / f for f in os.listdir(p)
            if any(f.lower().endswith(e) for e in targets)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No {fmt} file found in {p}"
            )
        file_path = str(candidates[0])
        if len(candidates) > 1:
            print(f"[{fmt.upper()}] Multiple files found — using {candidates[0].name}")
    else:
        file_path = str(p)

    print(f"[{fmt.upper()}] Loading {Path(file_path).name} ...")
    img = nib.load(file_path)
    data = np.array(img.dataobj, dtype=np.float32)

    print(f"[{fmt.upper()}] Raw shape from nibabel: {data.shape}")

    
    if data.ndim == 3:
        # Static volume — add T=1
        H, W, Z = data.shape
        volume = data.transpose(2, 0, 1)[np.newaxis]   
        volume = data.transpose(2, 0, 1)[np.newaxis]   

    elif data.ndim == 4:
        # Dynamic — (H, W, Z, T) → (T, Z, H, W)
        volume = data.transpose(3, 2, 0, 1)

    else:
        raise ValueError(
            f"Unexpected data dimensionality: {data.ndim}D. "
            "Expected 3D (static) or 4D (dynamic)."
        )

    T, Z, H, W = volume.shape
    print(f"[{fmt.upper()}] T={T}  Z={Z}  H={H}  W={W}")
    return volume




def _read_nrrd(input_path: str) -> np.ndarray:
    try:
        import nrrd
    except ImportError:
        raise ImportError(
            "pynrrd is required for NRRD files.\n"
            "Install it with:  pip install pynrrd"
        )

    p = Path(input_path)
    if p.is_dir():
        candidates = [p / f for f in os.listdir(p) if f.lower().endswith(".nrrd")]
        if not candidates:
            raise FileNotFoundError(f"No NRRD file found in {p}")
        file_path = str(candidates[0])
        if len(candidates) > 1:
            print(f"[NRRD] Multiple files found - using {candidates[0].name}")
    else:
        file_path = str(p)

    print(f"[NRRD] Loading {Path(file_path).name} ...")
    data, _header = nrrd.read(file_path)
    data = np.asarray(data, dtype=np.float32)
    print(f"[NRRD] Raw shape: {data.shape}")

    if data.ndim == 3:
        volume = data.transpose(2, 0, 1)[np.newaxis]   # (1, Z, H, W)
    elif data.ndim == 4:
        volume = data.transpose(3, 2, 0, 1)            # (T, Z, H, W)
    else:
        raise ValueError(
            f"Unexpected NRRD dimensionality: {data.ndim}D. "
            "Expected 3D (static) or 4D (dynamic)."
        )

    T, Z, H, W = volume.shape
    print(f"[NRRD] T={T}  Z={Z}  H={H}  W={W}")
    return volume


def read_any_mri(input_path: str) -> np.ndarray:
    """
    Load an MRI scan in any supported format and return (T, Z, H, W) float32.

    Parameters
    ----------
    input_path : str
        Path to a DICOM folder, a .nii/.nii.gz file, a .par file,
        a .nrrd file, a .hdr file, or a folder containing any of these.

    Returns
    -------
    np.ndarray, shape (T, Z, H, W), dtype float32
    """
    fmt = _detect_format(input_path)
    print(f"[MRI Reader] Detected format: {fmt.upper()}")

    if fmt == "dicom":
        return _read_dicom(input_path)

    if fmt == "nrrd":
        return _read_nrrd(input_path)

    if fmt in ("nifti", "parrec", "analyze"):
        return _read_nibabel(input_path, fmt)

    if fmt == "unknown_file":
        # Try nibabel first, fall back to DICOM
        try:
            import nibabel as nib
            img = nib.load(input_path)
            print("[MRI Reader] Unknown extension — nibabel opened it successfully.")
            return _read_nibabel(input_path, "nifti")
        except Exception:
            print("[MRI Reader] nibabel failed — trying DICOM reader.")
            return _read_dicom(input_path)

    raise ValueError(f"Unsupported format: {fmt}")
