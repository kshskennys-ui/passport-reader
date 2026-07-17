# Identity Document Data Page Extractor

Local desktop application for extracting normalized identity-document data pages from PDF, JPG, PNG, and TIFF inputs.

Stage 1 intentionally uses only OpenCV-based visual analysis. Phase 2A adds an optional PaddleOCR baseline over the immutable Stage 1 outputs; MRZ parsing, ICAO validation, and export remain out of scope.

## Run

```powershell
py -3.12 -m pip install -e ".[dev]"
py -3.12 src/main.py
```

The application saves standardized images, stage-by-stage debug artifacts, and low-confidence cases in the selected output directory.

## Phase 1B validation

Ground Truth lives under `dataset/`. Each label is independent of a pipeline segment number and records whether the source contains a data page, the required cardinal rotation, and the expected page count.

```powershell
py -3.13 scripts/generate_validation_report.py output/phase1b_baseline

py -3.13 scripts/run_regression.py `
  --label phase1b-v0.1 `
  --output-root output/phase1b-v0.1 `
  "C:\path\to\中联海防0704证件.pdf" `
  "C:\path\to\环球01证件.pdf"
```

Each regression produces `validation_report.html`, `validation_report.json`, category-specific `failed/` artifacts, and an appended `regression_history.json`. Use the in-app Review controls to confirm or reject a page result; every decision is saved in its dataset page directory as `review.json`.

To review an existing run without processing the PDF again, select the original input file, set the output directory to the saved run, and click `加载已有结果`. The GUI reconstructs the page table from `debug/*/page*/log.json`; Review decisions then update the report for every labeled source present in that output directory.

## Phase 1B.2 OCR-safety review

The normalizer is deliberately conservative:

- classification scores remain unchanged;
- segment crops expand into a configurable safe ROI;
- document ROIs expand again when dark content touches a detected boundary;
- normalization never trims after deskew;
- every output receives configurable white padding;
- deskew is applied only above the minimum angle and confidence thresholds, then retained only when a residual-angle measurement confirms improvement.

The Review panel records four independent quality decisions for every data page: page complete, portrait complete, MRZ complete, and OCR ready. Validation reports keep selection accuracy separate from crop completeness, MRZ completeness, portrait completeness, and OCR-ready rate. These quality rates remain `N/A` until reviewed; automatic edge warnings are diagnostic signals, not substitutes for human acceptance.

The current expanded result is under `output/expanded_samples_v0_8/`. It evaluates all 188 pages against visually audited Ground Truth and currently passes data-page presence/selection on all 188 pages. The 186 pages that contain data pages still require quality Review, so crop completeness, MRZ completeness, portrait completeness, and OCR-ready rate remain `N/A`.

## Phase 2A OCR baseline

Install the optional runtime and run OCR only against saved data-page images:

```powershell
py -3.13 -m pip install -e ".[ocr,dev]"
py -3.13 scripts/run_ocr_baseline.py `
  --input-root output/expanded_samples_v0_8/data_pages `
  --output-root output/ocr_baseline_v0_1
```

The baseline is resumable and never modifies Phase 1 images. It saves `ocr.json` and `overlay.png` for every page, plus `summary.json` and `ocr_report.html`. Analysis-only configuration changes reuse saved OCR text and regenerate metrics without running PaddleOCR again.

The current PP-OCRv5 mobile CPU run completed all 186 data pages with no runtime errors. Mean page confidence is 90.8%; 174 pages form MRZ candidate groups of at least two lines, while 12 pages have one MRZ line split across multiple OCR boxes. These are candidate-region diagnostics, not MRZ character accuracy or ICAO validation. Production field extraction should remain blocked until quality Review and MRZ-specific recognition validation are complete.
