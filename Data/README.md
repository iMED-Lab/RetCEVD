# Data format example

This directory contains one synthetic eye example: three 512 × 512 grayscale PNG images and `example.tsv`.

The images are artificial geometric patterns, not clinical OCTA scans. The label and all tabular values are placeholders. This example illustrates the input format, not training or performance evaluation.

`example.tsv` is UTF-8 encoded without a header and contains 72 tab-separated fields:

- `example_eye`: image directory relative to `Data/`.
- `0`: placeholder binary label.
- 70 placeholder tabular values, from which the loader selects 69 model inputs.

The three images use the filename suffixes expected by `Code/dataset.py`. Use `Data/` as the image root when reading this example. The single record is not a training/validation split.
