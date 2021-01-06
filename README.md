
# DeepFrag

This repository contains code for machine learning based lead optimization.

# Overview

- `config`: fixed configuration information (eg. TRAIN/VAL/TEST partitions)
- `configurations`: benchmark model configurations (see [`configurations/README.md`](configurations/README.md))
- `data`: training/inference data (see [`data/README.md`](data/README.md))
- `leadopt`: main module code
    - `models`: pytorch architecture definitions
    - `data_util.py`: utility code for reading packed fragment/fingerprint data files
    - `grid_util.py`: GPU-accelerated grid generation code
    - `metrics.py`: pytorch implementations of several metrics
    - `model_conf.py`: contains code to configure and train models
    - `util.py`: utility code for rdkit/openbabel processing
- `scripts`: data processing scripts (see [`scripts/README.md`](scripts/README.md))
- `train.py`: CLI interface to launch training runs

# Dependencies

You can build a virtualenv with the requirements:

```sh
$ python3 -m venv leadopt_env
$ source ./leadopt_env/bin/activate
$ pip install -r requirements.txt
```

Note: `Cuda 10.1` is required during training

# Training

To train a model, you can use the `train.py` utility script. You can specify model parameters as command line arguments or load parameters from a configuration args.json file.

```bash
python train.py \
    --save_path=/path/to/model \
    --wandb_project=my_project \
    {model_type} \
    --model_arg1=x \
    --model_arg2=y \
    ...
```

or

```bash
python train.py \
    --save_path=/path/to/model \
    --wandb_project=my_project \
    --configuration=./configurations/args.json
```

`save_path` is a directory to save the best model. The directory will be created if it doesn't exist. If this is not provided, the model will not be saved.

`wandb_project` is an optional wandb project name. If provided, the run will be logged to wandb.

See below for available models and model-specific parameters:

# Leadopt Models

In this repository, trainable models are subclasses of `model_conf.LeadoptModel`. This class encapsulates model configuration arguments and pytorch models and enables saving and loading multi-component models.

```py
from leadopt.model_conf import LeadoptModel, MODELS

model = MODELS['voxel']({args...})
model.train(save_path='./mymodel')

...

model2 = LeadoptModel.load('./mymodel')
```

Internally, model arguments are configured by setting up an `argparse` parser and passing around a `dict` of configuration parameters in `self._args`.

## VoxelNet

```
--no_partitions     If set, disable the use of TRAIN/VAL partitions during
                    training.
-f FRAGMENTS, --fragments FRAGMENTS
                    Path to fragments file.
-fp FINGERPRINTS, --fingerprints FINGERPRINTS
                    Path to fingerprints file.
-lr LEARNING_RATE, --learning_rate LEARNING_RATE
--num_epochs NUM_EPOCHS
                    Number of epochs to train for.
--test_steps TEST_STEPS
                    Number of evaluation steps per epoch.
-b BATCH_SIZE, --batch_size BATCH_SIZE
--grid_width GRID_WIDTH
--grid_res GRID_RES
--fdist_min FDIST_MIN
                    Ignore fragments closer to the receptor than this
                    distance (Angstroms).
--fdist_max FDIST_MAX
                    Ignore fragments further from the receptor than this
                    distance (Angstroms).
--fmass_min FMASS_MIN
                    Ignore fragments smaller than this mass (Daltons).
--fmass_max FMASS_MAX
                    Ignore fragments larger than this mass (Daltons).
--ignore_receptor
--ignore_parent
-rec_typer {single,single_h,simple,simple_h,desc,desc_h}
-lig_typer {single,single_h,simple,simple_h,desc,desc_h}
-rec_channels REC_CHANNELS
-lig_channels LIG_CHANNELS
--in_channels IN_CHANNELS
--output_size OUTPUT_SIZE
--pad
--blocks BLOCKS [BLOCKS ...]
--fc FC [FC ...]
--use_all_labels
--dist_fn {mse,bce,cos,tanimoto}
--loss {direct,support_v1}
```
