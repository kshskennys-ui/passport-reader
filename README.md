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

## Phase 2B MRZ second pass

The targeted MRZ experiment reads the existing OCR geometry, clusters split fragments into a safe lower-page band, and runs a second OCR pass on that band:

```powershell
py -3.13 scripts/run_mrz_second_pass.py `
  --baseline-root output/ocr_baseline_v0_1 `
  --output-root output/ocr_mrz_v0_1
```

The default run processes only pages where the first pass reported an incomplete MRZ candidate. It writes `mrz_region_overlay.png`, `mrz_crop.png`, `mrz_ocr_overlay.png`, and `mrz.json` per page. The current targeted run located all 12 MRZ bands with no crop-region failure. The crop images show that the MRZ itself is present, so remaining failures are OCR line-detection issues rather than Phase 1 crop loss.

## Phase 2C MRZ row recognition

Phase 2C keeps the Phase 2B safe band and additionally crops each located row, enlarges it, runs OCR independently, then merges fragments from left to right. It writes `mrz_row_01.png`, `mrz_row_01_ocr_overlay.png`, and corresponding files for each located row. The report includes merged text length and whether the recognized row lengths are equal. These are recognition diagnostics only; they do not perform field parsing, checksum validation, or automatic character correction.

```powershell
py -3.13 scripts/run_mrz_second_pass.py `
  --baseline-root output/ocr_baseline_v0_1 `
  --output-root output/ocr_mrz_v0_4 `
  --no-resume
```

## Phase 3 MRZ parsing and validation

The parser consumes saved row-level OCR results without rerunning PaddleOCR. It reconstructs concatenated rows, detects TD1/TD2/TD3 layouts, parses TD3 passport fields, and applies ICAO modulus-10 check digits with the `7-3-1` weighting. Recovery is limited to structural filler/marker repairs, blank optional-data check digits, and a small OCR-confusion set; a candidate is accepted only when all TD3 check digits pass, and the recovery method is recorded in `mrz.json`.

```powershell
py -3.13 scripts/run_mrz_parser.py `
  --input-root output/ocr_mrz_group_all_crew_v0_1 `
  --all-pages
```

The report separates valid parses from invalid, incomplete, and unsupported-format results. A valid parse is a structural and checksum result, not a guarantee that the source document identity has been independently verified.

## Fast MRZ-first OCR

For production-style MRZ extraction, use the fast runner directly against Phase 1 data-page images:

```powershell
py -3.13 scripts/run_fast_mrz.py `
  --input-root output/phase1/data_pages `
  --output-root output/fast_mrz_v0_1 `
  --workers 2
```

The fast path OCRs only the lower page band first. Pages that pass MRZ parsing do not run full-page OCR or row-level OCR. Pages that fail are automatically retried with full-page MRZ localization and, if needed, row-level OCR. Debug output is retained for the fast band, fallback crop, overlays, and any fallback row passes.

`--workers 2` processes different pages in separate processes, with one PaddleOCR model per process. On the current machine, it reduced a 19-page sample from about 305 seconds to 188 seconds, with about 1.8 GB model memory. Use `--workers 1` when diagnosing a single page; 3 workers are available but are not the default because the additional gain is small compared with the extra memory.

For targeted rechecks, add page numbers after `--pages`, for example `--pages 7 10 26`. The report distinguishes MRZ-region location from final MRZ parsing, and records fast-band versus full-page fallback counts separately.

The existing `run_mrz_second_pass.py` also now skips row-level OCR when the targeted MRZ crop already produces a valid parse. Use the fast runner for new files; keep the baseline plus second-pass workflow for diagnostic comparisons and legacy outputs.

## Redacted examples

The `examples/` directory contains sanitized screenshots showing the normalized data page and MRZ localization overlay. Portraits, names, document numbers, dates, and MRZ characters are irreversibly masked; the original source files and runtime outputs are excluded from Git.
