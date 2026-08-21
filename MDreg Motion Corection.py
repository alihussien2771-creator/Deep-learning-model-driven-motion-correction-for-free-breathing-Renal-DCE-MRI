from __future__ import annotations
import os
import mdreg
import json
from skimage.metrics import structural_similarity as ssim
from pydicom.uid import ExplicitVRLittleEndian
import pydicom
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import time
import logging


def _get_n_workers() -> int:

    for var in ("SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "OMP_NUM_THREADS"):
        v = os.environ.get(var)
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return os.cpu_count() or 1


N_WORKERS: int = _get_n_workers()
N_SCANS_PARALLEL: int = 2


_ONE = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = _ONE
os.environ["OMP_NUM_THREADS"] = _ONE
os.environ["OPENBLAS_NUM_THREADS"] = _ONE
os.environ["MKL_NUM_THREADS"] = _ONE

# ============================================================

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


# ============================================================
#  USER SETTINGS & PATHS
# ============================================================

MODE = ""    # "AIF_check" | "test" | "MID_TEST" | "full" | "view"
AIF_KNOWN = False   # True  -> load saved AIF (skip Napari)
# False -> open Napari to draw aorta first
VIEW_TYPE = ""  # "compare" (before+after) | "after" | "off"

# Root folder; one sub-folder per scan
INPUT_DIR = r""
# Results are written here
OUTPUT_DIR = r""

# Spatial auto-cropping: the registration runs on a tight crop of the
# anatomy to avoid wasting computation on background voxels.
CROP_THRESHOLD = 0.05   # Background cutoff as a fraction of the image maximum
# Extra pixels added around the bounding box (prevents edge artefacts)
CROP_PAD = 10

# ============================================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ============================================================
#  LOGGING SETUP
# ============================================================


def setup_logging(output_root: Path) -> None:

    output_root.mkdir(parents=True, exist_ok=True)
    log_file = output_root / "pipeline.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.info(f"Logging to {log_file}")


def _progress(iterable, **kwargs):

    if _TQDM_AVAILABLE:
        return _tqdm(iterable, **kwargs)
    return iterable


# ============================================================
#  DICOM I/O
#  Reads raw DICOM files, extracts key metadata tags, and
#  assembles the 4D image array (T, Z, H, W).
# ============================================================


def get_series_uid(ds: pydicom.Dataset) -> str:

    return str(ds.get("SeriesInstanceUID", ""))
# ===================================================
#  Extract Image Orientation
# Read the scanner's 6-number orientation tag to identify patient orientation
# and discard scout/localiser images with a different angle.
# ===================================================


def get_iop(ds: pydicom.Dataset) -> tuple | None:

    iop = ds.get("ImageOrientationPatient", None)
    return tuple(round(float(v), 6) for v in iop) if iop else None
# ============================================================
#  SLICE POSITION READER
#  Extracts the Z coordinate (height in mm) from each DICOM
#  file to group slices by their physical position in the body.
# ============================================================


def get_z(ds: pydicom.Dataset) -> float:

    ipp = ds.get("ImagePositionPatient", None)
    return float(ipp[2]) if (ipp and len(ipp) >= 3) else 0.0

# ============================================================
#  ACQUISITION TIME READER
#  Reads the time each image was taken from the DICOM tag and
#  converts it to total seconds since midnight for easy sorting.
# ============================================================


def _parse_acq_time(ds: pydicom.Dataset) -> float | None:

    t = ds.get("AcquisitionTime", None)
    if t is None:
        return None
    try:
        t_str = str(t).strip().replace(":", "")
        if len(t_str) < 4:
            return None
        h = int(t_str[0:2])
        m = int(t_str[2:4])
        s = float(t_str[4:]) if len(t_str) > 4 else 0.0
        return h * 3600 + m * 60 + s
    except Exception:
        return None
# ============================================================
# FRAME SORTER
# Determines the acquisition order of each DICOM slice using
# three fallback options (TemporalPositionIdentifier, AcquisitionTime, InstanceNumber )
# to handle differences between MRI scanner manufacturers.
# ============================================================


def get_time(ds: pydicom.Dataset) -> float | int:

    t = ds.get("TemporalPositionIdentifier", None)
    if t is not None:
        return int(t)
    acq = _parse_acq_time(ds)
    if acq is not None:
        return acq
    return int(ds.get("InstanceNumber", 0))

# ============================================================
#  TEMPORAL RESOLUTION ESTIMATOR
#  Estimates the time gap between frames in seconds.
#  Three  methods in order of reliability:
#   1. Read TemporalResolution tag directly from DICOM
#   2. Calculate median gap between AcquisitionTime values in the Z-slices
#   3. Default to 3.0 s if both methods fail
# ============================================================


def get_temporal_resolution_seconds(dicoms: list) -> float:

    for ds in dicoms:
        val = ds.get("TemporalResolution", None)
        if val is not None:
            try:
                return float(val) / 1000.0
            except Exception:
                pass

    z_groups: dict = defaultdict(list)
    for ds in dicoms:
        z_groups[round(get_z(ds), 1)].append(ds)

    for _, group in sorted(z_groups.items()):
        times = sorted(filter(lambda x: x is not None,
                              [_parse_acq_time(ds) for ds in group]))
        if len(times) >= 2:
            diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
            median_diff = float(np.median(diffs))
            if 0 < median_diff < 3600:
                logging.info(
                    f"Temporal resolution derived from AcquisitionTime: {median_diff:.2f} s")
                return median_diff
        break

    logging.warning(
        "Could not determine temporal resolution from DICOM. Defaulting to 3.0 s.")
    return 3.0
# ============================================================
#  DICOM FILE READER
#  Scans the patient folder, loads all valid DICOM files,
#  and filters them down to the real scan only.
# ============================================================


def read_dicoms(folder_path: Path) -> list:

    dicoms = []
    for p in folder_path.rglob('*'):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p, force=True)
            if hasattr(ds, "PixelData") or (0x7FE0, 0x0010) in ds:
                dicoms.append(ds)
        except Exception:
            continue
    if not dicoms:
        raise RuntimeError(
            f"No DICOMs found in {folder_path} or its subfolders")

    # Keep only the dominant series (by file count).
    # Skip filter if all files have unique SeriesInstanceUIDs (e.g. Siemens .IMA),
    # as applying it would incorrectly reduce the dataset to 1 file.
    series_counts = Counter(get_series_uid(d) for d in dicoms)
    top_series, top_count = series_counts.most_common(1)[0]
    if top_count > 1:
        dicoms = [d for d in dicoms if get_series_uid(d) == top_series]

    # Keep only the dominant image orientation (removes localisers)
    iop_counts = Counter(get_iop(d) for d in dicoms)
    main_iop = iop_counts.most_common(1)[0][0]
    return [d for d in dicoms if get_iop(d) == main_iop]


# ============================================================
#  ARRAY CONSTRUCTION & SPATIAL UTILITIES
# ============================================================


# ============================================================
#  4D ARRAY BUILDER
#  Assembles all DICOM files into one organised 4D array
#  with shape (T, Z, H, W) ready for motion correction.
# ============================================================


def build_stack(dicoms: list) -> np.ndarray:

    buckets: dict = defaultdict(list)
    for ds in dicoms:
        buckets[round(get_z(ds), 1)].append(ds)
    z_vals = sorted(buckets.keys())
    for z in z_vals:
        buckets[z].sort(key=get_time)

    counts = {z: len(buckets[z]) for z in z_vals}
    T = min(counts.values())
    max_T = max(counts.values())
    if T != max_T:
        logging.warning(
            f"Inconsistent timeframe counts across slices "
            f"(min={T}, max={max_T}). Truncating all slices to {T} timeframes. "
            "This usually indicates a DICOM sorting issue -- verify your data.")

    Z = len(z_vals)
    H, W = buckets[z_vals[0]][0].pixel_array.shape
    stack = np.zeros((T, Z, H, W), dtype=np.float32)
    for zi, z in enumerate(z_vals):
        for ti in range(T):
            stack[ti, zi] = buckets[z][ti].pixel_array
    return stack

# ============================================================
#  SPATIAL AUTO-CROPPER
#  Automatically removes the air surrounding the patient by
#  finding the brightest pixel in each position ever reached
#  across all frames and slices , then drawing a tight
#  bounding box around the body with a  safety border.
#  Returns the cropped 4D array and the box coordinates so
#  the result can be placed back into the full image later.
# ============================================================


def auto_crop_spatial(
    stack: np.ndarray,
    pad: int = CROP_PAD,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:

    mip = np.max(stack, axis=(0, 1))
    mask = mip > (CROP_THRESHOLD * mip.max())
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        # Entire image is background (e.g. corrupted DICOM) -- use full extent
        logging.warning(
            "auto_crop_spatial: no foreground found; using full image.")
        return stack, (0, mip.shape[0], 0, mip.shape[1])
    rmin = max(0, np.where(rows)[0][0] - pad)
    rmax = min(mip.shape[0], np.where(rows)[0][-1] + pad)
    cmin = max(0, np.where(cols)[0][0] - pad)
    cmax = min(mip.shape[1], np.where(cols)[0][-1] + pad)
    return stack[:, :, rmin:rmax, cmin:cmax], (rmin, rmax, cmin, cmax)


# ============================================================
#  REPORTING & QC OUTPUT
# ============================================================


def save_mp4(array_4d: np.ndarray, path: Path, fps: int = 8) -> None:

    import imageio

    T, Z = array_4d.shape[:2]
    z_mid = Z // 2
    frames = array_4d[:, z_mid, :, :]          

    
    vmin, vmax = float(frames.min()), float(frames.max())
    denom = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
    frames_u8 = ((frames - vmin) / denom * 255).astype(np.uint8)

    
    rgb_frames = [np.stack([f, f, f], axis=-1) for f in frames_u8]

    imageio.mimwrite(str(path), rgb_frames, fps=fps, macro_block_size=1)
    logging.info(f"Saved mp4: {path.name}")


def write_report(
    out_dir: Path,
    time_sec: float,
    before: np.ndarray,
    after: np.ndarray,
    mode: str,
) -> None:

    data_range = float(after.max() - after.min())
    T, Z = before.shape[0], before.shape[1]
    ssim_scores = []
    for t in range(T):
        for z in range(Z):
            s, _ = ssim(before[t, z], after[t, z],
                        data_range=data_range, full=True)
            ssim_scores.append(s)
    similarity = float(np.mean(ssim_scores))
    with open(out_dir / "processing_report.txt", "w") as f:
        f.write(
            f"=== MRI Motion Correction Report ({mode.upper()} MODE) ===\n")
        f.write(f"Calculation Time: {time_sec:.2f}s\n")
        f.write(
            f"Global SSIM (mean over {T}T x {Z}Z slices): {similarity:.4f}\n")
    logging.info("Report saved.")


def open_napari(before_path: Path, after_path: Path, v_type: str) -> None:

    import napari
    viewer = napari.Viewer()
    if v_type == "compare":
        viewer.add_image(np.load(before_path), name="Before", colormap="gray")
    viewer.add_image(np.load(after_path), name="After", colormap="gray")
    napari.run()


def save_corrected_dicoms(
    corrected_stack: np.ndarray,
    source_dicoms: list,
    output_dir: Path,
) -> None:

    dicom_out_dir = output_dir / "corrected_dicoms"
    dicom_out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict = defaultdict(list)
    for ds in source_dicoms:
        buckets[round(get_z(ds), 1)].append(ds)
    z_vals = sorted(buckets.keys())
    for z in z_vals:
        buckets[z].sort(key=get_time)

    T = corrected_stack.shape[0]
    dicom_t = len(buckets[z_vals[0]]) if z_vals else 0
    if T > dicom_t:
        logging.warning(
            f"save_corrected_dicoms: corrected stack has {T} timeframes but only "
            f"{dicom_t} source DICOMs available. Saving {dicom_t} timeframe(s).")
        T = dicom_t
    global_min = float(corrected_stack.min())
    global_max = float(corrected_stack.max())
    slope = (global_max - global_min) / \
        65535.0 if global_max > global_min else 1.0

    saved = 0
    for zi, z in enumerate(z_vals):
        for ti in range(T):
            ds = buckets[z][ti]
            corrected_2d = corrected_stack[ti, zi]
            pixel_u16 = (
                (corrected_2d - global_min) / slope
            ).round().clip(0, 65535).astype(np.uint16)

            ds.PixelData = pixel_u16.tobytes()
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.RescaleSlope = slope
            ds.RescaleIntercept = global_min

            try:
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            except AttributeError:
                pass

            fname = dicom_out_dir / f"slice{zi:03d}_time{ti:03d}.dcm"
            ds.save_as(str(fname))
            saved += 1

    logging.info(f"Saved {saved} corrected DICOM files to {dicom_out_dir}")


# ============================================================
#  WORKER SAFETY INITIALISER
#  Runs once inside each worker process at startup.
#  Resets all library thread limits to 1 to prevent each
#  worker from grabbing cores that belong to other workers.
# ============================================================


def _worker_init() -> None:

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
# ============================================================
#  MOTION CORRECTION ENGINE
#  The core worker function that processes one Z-slice at a time.
# Runs in parallel across all CPU cores.
# ============================================================


def _load_ants_transfo(transfo) -> np.ndarray | None:
    # transfo shape from mdreg ANTs: (T, 2) of string paths.
    # col 0 = Warp.nii.gz (displacement field), col 1 = GenericAffine.mat (not needed).
    # Returns (H, W, T, 2) float64 array matching the correct fields.npy format.
    try:
        import ants as _ants
        arr = np.asarray(transfo)
        if arr.dtype.kind not in ('U', 'S'):
            return transfo  
        T = arr.shape[0]
        warp_col = arr[:, 0] if arr.ndim == 2 else arr  
        warp_arrays = []
        for t in range(T):
            path = str(warp_col[t])
            warp = _ants.image_read(path).numpy()  
            if warp.ndim == 4:
                warp = warp[:, :, 0, :]             
            warp_arrays.append(warp)
        
        return np.stack(warp_arrays, axis=2)
    except Exception as e:
        logging.warning(f"_load_ants_transfo failed: {e}")
        return None


def _process_slice(args: tuple) -> tuple:

    z, slice_2d_time, time_array, aif_array, max_iter = args

    
    slice_mdreg_input = np.transpose(slice_2d_time, (1, 2, 0))

    coreg, fitted, transfo, pars = mdreg.fit(
        slice_mdreg_input,
        fit_image={
            'func':        mdreg.fit_2cm_lin,  
            'time':        time_array,
            'aif':         aif_array,
            'baseline':    2,                  
            'input_corr':  True,               
        },
        fit_coreg={
            'package':           'ants',
            'type_of_transform': 'SyN',
            'parallel':          False,         
        },
        maxit=max_iter,
        verbose=0,
    )

    # Convert results back to (T, H, W) for consistent storage
  
    return (z,
            np.transpose(coreg,   (2, 0, 1)),
            np.transpose(fitted,  (2, 0, 1)),
            _load_ants_transfo(transfo),
            pars)


# ============================================================
#  ARTERIAL INPUT FUNCTION (AIF) EXTRACTION
#  The AIF is the  concentration curve in the aorta.
#  It is required by the 2CM-Lin pharmacokinetic model to
#  separate tissue perfusion from bulk motion.
# ============================================================


def extract_single_aif(stack_4d: np.ndarray, scan_name: str = "Current Scan") -> np.ndarray:

    import napari
    mip_3d = np.max(stack_4d, axis=0)  # collapse T -> (Z, H, W) projection
    viewer = napari.Viewer(
        title=f"Draw Aorta for: {scan_name} (Then close window)")
    viewer.add_image(mip_3d, name=f"MIP - {scan_name}")
    labels_layer = viewer.add_labels(
        np.zeros_like(mip_3d, dtype=int), name="Draw Aorta Here")

    logging.info(
        f"\n*** WAITING FOR USER INPUT: {scan_name} ***\n"
        "1. Find the Aorta.\n2. Paint a small label.\n3. Close the window.")
    napari.run()

    mask_3d = labels_layer.data == 1
    if not np.any(mask_3d):
        raise ValueError(
            f"No label drawn for {scan_name}! "
            "Please paint the aorta before closing the window.")

    time_frames = stack_4d.shape[0]
    curve_1d = np.array([np.mean(stack_4d[t, mask_3d])
                        for t in range(time_frames)])
    return curve_1d


# ============================================================
#  MAIN ENGINE
#  Orchestrates the full batch pipeline: iterates over all
#  scan folders, handles all modes, manages I/O, and
#  dispatches parallel slice workers.
# ============================================================


def _run_patient(args: tuple) -> None:
    i, total, patient_dir, output_root, n_parallel = args

    logging.info(f"\n{'='*50}")
    logging.info(f"Processing Scan {i+1}/{total}: {patient_dir.name}")
    logging.info(f"{'='*50}")

    patient_out_dir = output_root / patient_dir.name
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    before_f = patient_out_dir / "stack4d_before.npy"
    after_f   = patient_out_dir / "stack4d_after.npy"
    fit_f     = patient_out_dir / "stack4d_fit.npy"
    fields_f  = patient_out_dir / "fields.npy"
    aif_file  = patient_out_dir / "aif_curve.npy"

    # -- 1. AIF_CHECK MODE ---------------------------------------------
    if MODE == "AIF_check":
        if aif_file.exists():
            logging.info(f"AIF already exists for {patient_dir.name}. Skipping.")
            return
        try:
            logging.info("Reading DICOMs...")
            dicoms = read_dicoms(patient_dir)
            stack_f32 = build_stack(dicoms)
            aif_array = extract_single_aif(stack_f32, scan_name=patient_dir.name)
            np.save(aif_file, aif_array)
            logging.info(f"Saved AIF to {aif_file}")
        except Exception as e:
            logging.error(f"Failed to process {patient_dir.name}: {e}")
        return

    # -- 2. VIEW MODE --------------------------------------------------
    if MODE == "view":
        if VIEW_TYPE != "off" and before_f.exists() and after_f.exists():
            logging.info("Opening Napari to view results...")
            open_napari(before_f, after_f, VIEW_TYPE)
        else:
            logging.warning(f"Cannot view {patient_dir.name} -- missing .npy files.")
        return

    # -- 3. MOTION CORRECTION (test / MID_TEST / full) -----------------
    try:
        logging.info("Reading DICOMs...")
        dicoms = read_dicoms(patient_dir)
        stack_f32 = build_stack(dicoms)
    except Exception as e:
        logging.error(f"Failed to read DICOMs for {patient_dir.name}: {e}")
        return

    try:
        time_frames_total = stack_f32.shape[0]
        tr_seconds = get_temporal_resolution_seconds(dicoms)
        time_array = np.arange(time_frames_total) * tr_seconds
        logging.info(
            f"Temporal resolution: {tr_seconds:.2f} s  |  Timeframes: {time_frames_total}")

        if AIF_KNOWN:
            if not aif_file.exists():
                logging.warning(
                    f"AIF_KNOWN is True but no saved AIF found for "
                    f"{patient_dir.name}. Skipping.")
                return
            aif_array = np.load(aif_file)
        else:
            logging.info("Opening Napari to extract AIF curve...")
            aif_array = extract_single_aif(stack_f32, scan_name=patient_dir.name)
            np.save(aif_file, aif_array)

        if len(aif_array) != time_frames_total:
            logging.error(
                f"AIF length ({len(aif_array)}) does not match stack timeframes "
                f"({time_frames_total}) for {patient_dir.name}. "
                "Delete aif_curve.npy and re-extract. Skipping.")
            return

        Z_slices = stack_f32.shape[1]
        z_loop = list(range(Z_slices))

        if MODE == "full":
            logging.info("FULL MODE: Processing entire 4D stack.")
            max_iter = 5
        elif MODE == "MID_TEST":
            logging.info("MID_TEST MODE: Full timeline, middle slice only.")
            max_iter = 3
            z_loop = [Z_slices // 2]
        elif MODE == "test":
            logging.info("TEST MODE: Truncating to 10 timeframes.")
            t_limit = min(10, stack_f32.shape[0])
            stack_f32 = stack_f32[:t_limit]
            aif_array = aif_array[:t_limit]
            time_array = time_array[:t_limit]
            max_iter = 1

        np.save(before_f, stack_f32)

        cropped_stack, bounds = auto_crop_spatial(stack_f32)

        logging.info(
            f"Starting mdreg optimisation -- {len(z_loop)} slice(s), "
            f"{n_parallel} worker(s), max_iter={max_iter}...")
        start_time = time.time()
        after_f32 = np.copy(stack_f32)

        tasks = [
            (z, cropped_stack[:, z, :, :].copy(), time_array, aif_array, max_iter)
            for z in z_loop
        ]

        results: dict = {}
        with ProcessPoolExecutor(max_workers=n_parallel,
                                 initializer=_worker_init) as pool:
            future_to_z = {pool.submit(_process_slice, task): task[0]
                           for task in tasks}
            for future in _progress(as_completed(future_to_z),
                                    total=len(future_to_z),
                                    desc=f"{patient_dir.name} slices",
                                    unit="slice"):
                z, coreg_t, fit_t, transfo, pars = future.result()
                results[z] = (coreg_t, fit_t, transfo, pars)
                logging.info(f"Slice {z + 1}/{Z_slices} done.")

        T_frames = stack_f32.shape[0]
        H_full   = stack_f32.shape[2]
        W_full   = stack_f32.shape[3]
        fit_4d   = np.zeros_like(stack_f32)
        # Pre-allocate full-size fields array: (Z, H, W, T, 2) matching correct format
        fields_5d = np.zeros((Z_slices, H_full, W_full, T_frames, 2), dtype=np.float64)
        all_pars = []
        for z in z_loop:
            coreg_t, fit_t, transfo, pars = results[z]
            after_f32[:, z, bounds[0]:bounds[1], bounds[2]:bounds[3]] = coreg_t
            fit_4d[:, z,    bounds[0]:bounds[1], bounds[2]:bounds[3]] = fit_t
            if transfo is not None:
                fields_5d[z, bounds[0]:bounds[1], bounds[2]:bounds[3], :, :] = transfo
            else:
                logging.warning(f"transfo is None for slice {z} -- fields will be zeros for this slice.")
            all_pars.append((z, pars))

        np.save(after_f,  after_f32)
        np.save(fit_f,    fit_4d)
        try:
            np.save(fields_f, fields_5d)
        except Exception as e:
            logging.warning(f"Could not save fields.npy for {patient_dir.name}: {e}")
        np.savez(patient_out_dir / "param_maps.npz",
                 **{f"slice_{z:03d}": pars for z, pars in all_pars})

        with open(patient_out_dir / "crop_bounds.json", "w") as f:
            json.dump({
                "rmin": int(bounds[0]), "rmax": int(bounds[1]),
                "cmin": int(bounds[2]), "cmax": int(bounds[3]),
            }, f)

        write_report(patient_out_dir, time.time() - start_time,
                     stack_f32, after_f32, MODE)

        logging.info("Saving mp4 previews...")
        save_mp4(stack_f32, patient_out_dir / "preview_before.mp4")
        save_mp4(after_f32, patient_out_dir / "preview_after.mp4")
        save_mp4(fit_4d,    patient_out_dir / "preview_fit.mp4")

        save_corrected_dicoms(after_f32, dicoms, patient_out_dir)

        if VIEW_TYPE != "off":
            open_napari(before_f, after_f, VIEW_TYPE)

    except Exception as e:
        logging.error(
            f"Failed to process {patient_dir.name}: {e}", exc_info=True)


def main() -> None:

    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)

    if not input_root.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_root}")

    setup_logging(output_root)

    patient_folders = [f for f in input_root.iterdir() if f.is_dir()]
    total = len(patient_folders)
    logging.info(f"Found {total} patient(s)/scan(s) in {INPUT_DIR}.")

    # Napari/Qt must run on the main thread. Process GUI-enabled modes
    # sequentially; only headless processing may use the scan thread pool.
    if MODE == "AIF_check" or VIEW_TYPE != "off":
        args_list = [(i, total, p, output_root, N_WORKERS) for i, p in enumerate(patient_folders)]
        for args in args_list:
            _run_patient(args)
    else:
        n_parallel = max(1, N_WORKERS // min(N_SCANS_PARALLEL, total))
        logging.info(f"Cores per scan: {n_parallel}  |  Concurrent scans: {min(N_SCANS_PARALLEL, total)}")
        args_list = [(i, total, p, output_root, n_parallel) for i, p in enumerate(patient_folders)]
        with ThreadPoolExecutor(max_workers=N_SCANS_PARALLEL) as executor:
            list(executor.map(_run_patient, args_list))


# -- Entry point ---------------------------------------------------------------
# The main() function is called inside this guard to allow safe multiprocessing
# to prevent each worker process from re-executing the top-level script
if __name__ == "__main__":
    main()

