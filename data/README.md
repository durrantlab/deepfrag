This folder contains data used during training and inference.

Model configuration files in `/configurations` expect the data files to be in this directory. You can either copy them directly here or use symlinks.

You can download the data here: https://pitt.box.com/s/ubohnl10idnarpam40hq6chggtaojqv7

Overview:
- `moad.h5` (7 GB): processed MOAD data loaded by `data_util.FragmentDataset`
- `rdk10_moad` (384 MB): RDK-10 fingerprints for MOAD data loaded by `data_util.FingerprintDataset` (generated with `scripts/make_fingerprints.py`)
