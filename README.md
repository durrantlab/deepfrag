
# Lead Optimization via Fragment Prediction

# Overview

- `config`: configuration information (eg. TRAIN/TEST partitions)
- `data`: training/inference data (see [`data/README.md`](data/README.md))
- `docker`: Docker environment
- `leadopt`: main module code
    - `models`: architecture definitions
    - `data_util.py`: utility wrapper code around fragment and fingerprint datasets
    - `grid_util.py`: GPU-accelerated grid generation code
    - `infer.py`: code for inference with a trained model
    - `metrics.py`
    - `train.py`: training loops
    - `util.py`: extra utility code (mostly rdkit)
- `pretained`: pretrained models (see [`pretrained/README.md`](pretrained/README.md))
- `scripts`: data processing scripts (see [`scripts/README.md`](scripts/README.md))
- `train.py`: CLI interface to launch training runs
- `leadopt.py`: CLI interface to run inference on new samples

# Training

You can train models with the `train.py` utility script
