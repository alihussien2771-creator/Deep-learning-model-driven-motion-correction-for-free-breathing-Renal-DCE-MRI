# Renal DCE-MRI Motion Correction

This repository contains two motion-correction pipelines for renal dynamic contrast-enhanced MRI (DCE-MRI):

1. `MDreg Motion Corection.py` generates model-driven registration (MDR) reference outputs from raw DICOM scans.
2. `Deep Learning Motion Correction.py` trains or runs the V13 deep-learning model, a group-referenced cascaded U-Net that approximates MDR-like in-plane motion correction.

The repository also includes the selected V13 checkpoint: `best_model_cascade3.pth`.

This code is intended for research use. It is not a validated clinical product.

## Repository files

- `MDreg Motion Corection.py`
- `Deep Learning Motion Correction.py`
- `dicom_to_npy.py`
- `mri_reader.py`
- `best_model_cascade3.pth`
- `README.md`

The filenames contain spaces, so run them in quotes. For example:

- `python ".\Deep Learning Motion Correction.py"`
- `python ".\MDreg Motion Corection.py"`

## Expected data format

The MDR pipeline reads raw DICOM folders and exports NumPy arrays. The deep-learning pipeline can use either prepared NumPy scan folders or raw DICOM folders.

For training, the deep-learning pipeline expects one folder per scan with:

- `stack4d_before.npy`: uncorrected input series, shape `(T, Z, H, W)`
- `stack4d_after.npy`: MDR-corrected target, shape `(T, Z, H, W)`
- `fields.npy`: MDR displacement fields, usually shape `(Z, H, W, T, 2)`

For inference with the trained V13 checkpoint, there are two options:

- Prepared NumPy input: each scan folder contains `stack4d_before.npy`.
- Raw DICOM input: the script uses `dicom_to_npy.py` and `mri_reader.py` to convert raw DICOM folders into `stack4d_before.npy` first.

The model was developed for renal DCE-MRI stacks with 265 temporal frames, 8 slices and a 384 × 384 reconstructed in-plane matrix. The network internally resizes images to 192 × 192 for prediction, then returns outputs at the original resolution.

## Environment

Use Python 3.10 or newer.

The deep-learning script requires:

- `numpy`
- `torch`
- `matplotlib`
- `imageio`

The MDR script additionally requires:

- `mdreg`
- `antspyx` / `ants`
- `pydicom`
- `scikit-image`
- `napari`
- `tqdm`
- `imageio`

Example environment setup:

- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `pip install numpy torch matplotlib imageio pydicom scikit-image tqdm napari antspyx`

Install `mdreg` according to the version/source used in your local MDR environment.

## 1. Generate MDR reference data

Open `MDreg Motion Corection.py` and edit the settings near the top of the file:

- `MODE = ""`
- `AIF_KNOWN = False`
- `VIEW_TYPE = ""`
- `INPUT_DIR = r""`
- `OUTPUT_DIR = r""`

`MODE` can be set to `"AIF_check"`, `"test"`, `"MID_TEST"`, `"full"` or `"view"`.

`VIEW_TYPE` can be set to `"compare"`, `"after"` or `"off"`.

`INPUT_DIR` should point to a folder containing one subfolder per scan. The script searches recursively inside each scan folder for valid DICOM files.

Common modes:

- `MODE = "AIF_check"` opens Napari to manually mark the aortic input function (AIF) and saves `aif_curve.npy`.
- `MODE = "test"` runs a short 10-frame test.
- `MODE = "MID_TEST"` runs the full timeline on the middle slice only.
- `MODE = "full"` runs the full MDR correction for all slices.
- `MODE = "view"` opens saved outputs for visual checking.

Typical AIF-extraction setup:

- `MODE = "AIF_check"`
- `AIF_KNOWN = False`
- `VIEW_TYPE = "off"`
- `INPUT_DIR = r"C:\path\to\raw_scan_root"`
- `OUTPUT_DIR = r"C:\path\to\mdr_output"`

Run:

- `python ".\MDreg Motion Corection.py"`

After AIF curves have been saved, run full MDR with:

- `MODE = "full"`
- `AIF_KNOWN = True`
- `VIEW_TYPE = "off"`
- `INPUT_DIR = r"C:\path\to\raw_scan_root"`
- `OUTPUT_DIR = r"C:\path\to\mdr_output"`

Run the script again:

- `python ".\MDreg Motion Corection.py"`

For each scan, the MDR pipeline writes outputs such as:

- `stack4d_before.npy`
- `stack4d_after.npy`
- `stack4d_fit.npy`
- `fields.npy`
- `param_maps.npz`
- `aif_curve.npy`
- `crop_bounds.json`
- `preview_before.mp4`
- `preview_after.mp4`
- `preview_fit.mp4`
- `processing_report.txt`

The files `stack4d_before.npy`, `stack4d_after.npy` and `fields.npy` are the key files used to train the deep-learning model.

## 2. Train V13 from MDR outputs

Open `Deep Learning Motion Correction.py` and edit:

- `MODE = "train"`
- `DATA_ROOT = r"C:\path\to\mdr_output"`
- `OUT_DIR = r"C:\path\to\v13_output"`

`DATA_ROOT` should contain one folder per scan. Each scan folder should contain:

- `stack4d_before.npy`
- `stack4d_after.npy`
- `fields.npy`

Run:

- `python ".\Deep Learning Motion Correction.py"`

Training writes:

- `best_model_cascade3.pth`
- `last_model_cascade3.pth`
- `training_cascade3.log`
- `training_history_cascade3.jsonl`

The included `best_model_cascade3.pth` is the selected trained V13 checkpoint.

## 3. Run V13 inference with the included checkpoint

For inference, use either prepared NumPy scan folders or raw DICOM scan folders.

By default, the script expects `best_model_cascade3.pth` to be inside `OUT_DIR`. Either copy the checkpoint into `OUT_DIR`, or edit `CHECKPOINT` to point to the checkpoint’s actual location.

Option A: prepared NumPy input

Prepare one or more scan folders containing `stack4d_before.npy`.

Then edit `Deep Learning Motion Correction.py`:

- `MODE = "infer"`
- `OUT_DIR = r"C:\path\to\v13_output"`
- `INFER_ROOT = r"C:\path\to\prepared_input_scans"`
- `CHECKPOINT = r"C:\path\to\best_model_cascade3.pth"`
- `INFER_OUT = os.path.join(OUT_DIR, "inference")`

Option B: raw DICOM input

Raw DICOM inference is also supported because this repository includes:

- `dicom_to_npy.py`
- `mri_reader.py`

For raw DICOM input, set `INFER_ROOT` to the raw scan root instead of a prepared NumPy folder:

- `MODE = "infer"`
- `OUT_DIR = r"C:\path\to\v13_output"`
- `INFER_ROOT = r"C:\path\to\raw_dicom_scan_root"`
- `CHECKPOINT = r"C:\path\to\best_model_cascade3.pth"`
- `AUTO_CONVERT_RAW = True`
- `RAW_PREPARED_DIR = os.path.join(OUT_DIR, "prepared_input")`

When no `stack4d_before.npy` file is found under `INFER_ROOT`, the script automatically calls `convert_batch` from `dicom_to_npy.py`. The converted inputs are saved under `RAW_PREPARED_DIR`, then V13 inference runs on those converted arrays.

Run:

- `python ".\Deep Learning Motion Correction.py"`

The output for each scan is written to `OUT_DIR/inference/scan_name/`.

Typical inference outputs are:

- `stack4d_before.npy`
- `stack4d_after.npy`
- `predicted_warped.npy`
- `predicted_fields.npy`
- `predicted_reference_working.npy`
- `predicted_trajectory.npy`
- `preview_before.mp4`
- `preview_after.mp4`
- `preview_reference.mp4`
- `contact_before_after_diff.png`
- `timecut_before.png`
- `timecut_after.png`
- `timecut_compare.png`

`stack4d_after.npy` and `predicted_warped.npy` both represent the V13-corrected image series. `predicted_fields.npy` contains the predicted in-plane displacement fields.

## 4. Run a quick model check

To check that the model, loss and data loading run on your machine, edit `Deep Learning Motion Correction.py`:

- `MODE = "check"`
- `DATA_ROOT = r"C:\path\to\mdr_output"`
- `OUT_DIR = r"C:\path\to\v13_output"`

Then run:

- `python ".\Deep Learning Motion Correction.py"`

This performs a small forward/backward pass on one scan and writes `check_cascade3.log`.

## Notes about raw DICOM conversion

Raw DICOM conversion is handled by `dicom_to_npy.py` and `mri_reader.py`; it is not performed by the neural network itself. During inference, `Deep Learning Motion Correction.py` checks `INFER_ROOT` as follows:

1. If `INFER_ROOT` itself contains `stack4d_before.npy`, it treats it as one prepared scan.
2. If subfolders under `INFER_ROOT` contain `stack4d_before.npy`, it treats them as multiple prepared scans.
3. If no prepared NumPy inputs are found and `AUTO_CONVERT_RAW = True`, it converts the raw scan folders into NumPy format using `dicom_to_npy.py`.

The raw converter searches recursively for valid DICOM files, including XNAT-style folders such as `resources/DICOM/files`.

The automatic conversion route produces prepared input folders containing `stack4d_before.npy` and `crop_bounds.json`. These prepared folders are then passed to the V13 model for correction.

## Model summary

V13 uses eight consecutive frames to calculate a group-mean reference image. Each frame is then registered separately to that reference using a three-stage cascaded U-Net. The model predicts two in-plane displacement components, `dx` and `dy`; it does not estimate through-plane motion.

The method is best described as group-referenced, frame-wise registration rather than fully joint groupwise registration.

## Important limitations

- The checkpoint approximates MDR outputs; MDR is the supervision target, not motion-free ground truth.
- The model predicts in-plane displacement only.
- The code has not been validated as a clinical device.
- Do not publish patient-identifiable DICOM files, derived private data or local paths in a public repository.
