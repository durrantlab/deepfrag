This folder contains data used during training and inference.

You can download the data here: https://pitt.box.com/s/ubohnl10idnarpam40hq6chggtaojqv7

Overview:
- `fragments_desc.h5` (3.9 GB): pdbbind fragment information created by `scripts/process_pdbbind2.py`
- `fragments_mini.h5` (824 KB): a minimal version of `fragments_desc.h5` with 3 receptors and 50 fragments for testing
- `fp_rdk_desc.h5` (625 MB): precomputed rdk fingerprints created by `scripts/make_fingerprints.py`
