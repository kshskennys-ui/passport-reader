# Ground Truth Dataset

Each source scan has one directory and an `expected.json` file. The labels describe source facts, not model candidate numbers, so they remain valid when segmentation changes.

`rotation` is the clockwise cardinal rotation required to make the source scan upright. `expected_pages` is the number of data pages present in the source scan.

`expected.png` is optional. It is created only after a reviewer confirms a normalized output is the correct canonical data-page image. This prevents a model prediction from being silently reused as its own ground truth.

The initial labels were created from visual audits of the supplied samples. Every later correction must retain the `remarks` field and be made through the Review flow or an explicit dataset update. Bootstrap scripts record their audit date and annotator in each label; pipeline predictions must never be copied into labels without source-image review.
