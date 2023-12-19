# DeepFrag

DeepFrag is a machine learning model for fragment-based lead optimization. In
this repository, you will find code to train the model and code to run
inference using a pre-trained model.

## Citation

If you use DeepFrag in your research, please cite as:

Green, H., Koes, D. R., & Durrant, J. D. (2021). DeepFrag: a deep
convolutional neural network for fragment-based lead optimization. Chemical
Science.

```tex
@article{green2021deepfrag,
  title={DeepFrag: a deep convolutional neural network for fragment-based lead optimization},
  author={Green, Harrison and Koes, David Ryan and Durrant, Jacob D},
  journal={Chemical Science},
  year={2021},
  publisher={Royal Society of Chemistry}
}
```

## Usage

There are three ways to use DeepFrag:

1. **DeepFrag Browser App**: We have released a free, open-source browser app
   for DeepFrag that requires no setup and does not transmit any structures to
   a remote server.
    - View the online version at
      [durrantlab.pitt.edu/deepfrag](https://durrantlab.pitt.edu/deepfrag/)
    - See the code at
      [git.durrantlab.pitt.edu/jdurrant/deepfrag-app](https://git.durrantlab.pitt.edu/jdurrant/deepfrag-app)
2. **DeepFrag CLI**: In this repository we have included a `deepfrag.py`
   script that can perform common prediction tasks using the API.
    - See the `DeepFrag CLI` section below
3. **DeepFrag API**: For custom tasks or fine-grained control over
   predictions, you can invoke the DeepFrag API directly and interface with
   the raw data structures and the PyTorch model. We have created an example
   Google Colab (Jupyter notebook) that demonstrates how to perform manual
   predictions.
    - See the interactive
      [Colab](https://colab.research.google.com/drive/1XWin26iDXqZ2ioGtwDRuO4iRomGVpdte).

## DeepFrag CLI

The DeepFrag CLI is invoked by running `python3 deepfrag.py` in this
repository. The CLI requires a pre-trained model and the fragment library to
run. You will be prompted to download both when you first run the CLI and
these will be saved in the `./.store` directory.

### Structure (specify exactly one)

The input structures are specified using either a manual receptor and ligand
pdb or by specifying a pdb id and the ligand residue number.

- `--receptor <rec.pdb> --ligand <lig.pdb>`
- `--pdb <pdbid> --resnum <resnum>`

### Connection Point (specify exactly one)

DeepFrag will predict new fragments that connect to the _connection point_ via
a single bond. You must specify the connection point atom using one of the
following:

- `--cname <name>`: Specify the connection point by atom name (e.g. `C3`,
  `N5`, `O2`, ...).
- `--cx <x> --cy <y> --cz <z>`: Specify the connection point by atomic
  coordinate. DeepFrag will find the closest atom to this point.

### Fragment Removal (optional) (specify exactly one)

If you are using DeepFrag for fragment _replacement_, you must first remove
the original fragment from the ligand structure. You can either do this by
hand, e.g. editing the PDB, or DeepFrag can do this for you by specifying
_which_ fragment should be removed.

_Note: predicting fragments in place of hydrogen atoms (e.g. protons) does not
require any fragment removal since hydrogen atoms are ignored by the model._

To remove a fragment, you specify a second atom that is contained in the
fragment. Like the connection point, you can either use the atom name or the
atom coordinate.

- `--rname <name>`: Specify the connection point by atom name (e.g. `C3`,
  `N5`, `O2`, ...).
- `--rx <x> --ry <y> --rz <z>`: Specify the connection point by atomic
  coordinate. DeepFrag will find the closest atom to this point.

### Output (optional)

By default, DeepFrag will print a list of fragment predictions to stdout
similar to the [Browser App](https://durrantlab.pitt.edu/deepfrag/).

- `--out <out.csv>`: Save predictions in CSV format to `out.csv`. Each line
  contains the fragment rank, score and SMILES string.

### Miscellaneous (optional)

- `--full`: Generate SMILES strings with the full ligand structure instead of
  just the fragment. (__IMPORTANT NOTE__: Bond orders are not assigned to the
  parent portion of the full ligand structure. These must be added manually.)
- `--cpu/--gpu`: DeepFrag will attempt to infer if a Cuda GPU is available and
  fallback to the CPU if it is not. You can set either the `--cpu` or `--gpu`
  flag to explicitly specify the target device.
- `--num_grids <num>`: Number of grid rotations to use. Using more will take
  longer but produce a more stable prediction. (Default: 4)
- `--top_k <k>`: Number of predictions to print in stdout. Use -1 to display
  all. (Default: 25)

## Reproduce Results

You can use the DeepFrag CLI to reproduce the highlighted results from the
main manuscript:

### 1. Fragment replacement

To replace fragments, specify the connection point (`cname` or `cx/cy/cz`) and
specify a second atom that is contained in the fragment (`rname` or
`rx/ry/rz`).

```bash
# Fig. 3: (2XP9) H. sapiens peptidyl-prolyl cis-trans isomerase NIMA-interacting 1 (HsPin1p)

# Carboxylate A
$ python3 deepfrag.py --pdb 2xp9 --resnum 1165 --cname C10 --rname C12

# Phenyl B
$ python3 deepfrag.py --pdb 2xp9 --resnum 1165 --cname C1 --rname C2

# Phenyl C
$ python3 deepfrag.py --pdb 2xp9 --resnum 1165 --cname C18 --rname C19
```

```bash
# Fig. 4A: (6QZ8) Protein myeloid cell leukemia1 (Mcl-1)

# Carboxylate group interacting with R263
$ python3 deepfrag.py --pdb 6qz8 --resnum 401 --cname C12 --rname C14

# Ethyl group
$ python3 deepfrag.py --pdb 6qz8 --resnum 401 --cname C6 --rname C10

# Methyl group
$ python3 deepfrag.py --pdb 6qz8 --resnum 401 --cname C25 --rname C30

# Chlorine atom
$ python3 deepfrag.py --pdb 6qz8 --resnum 401 --cname C28 --rname CL
```

```bash
# Fig. 4B: (1X38) Family GH3 b-D-glucan glucohydrolase (barley)

# Hydroxyl group interacting with R158 and D285
$ python3 deepfrag.py --pdb 1x38 --resnum 1001 --cname C2B --rname O2B

# Phenyl group interacting with W286 and W434
$ python3 deepfrag.py --pdb 1x38 --resnum 1001 --cname C7B --rname C1
```

```bash
# Fig. 4C: (4FOW) NanB sialidase (Streptococcus pneumoniae)

# Amino group
$ python3 deepfrag.py --pdb 4fow --resnum 701 --cname CAE --rname NAA
```

### 2. Fragment addition

For fragment addition, you only need to specify the atom connection point
(`cname` or `cx/cy/cz`). In this case, DeepFrag will implicitly replace a
valent hydrogen.

```bash
# Fig. 5: Ligands targeting the SARS-CoV-2 main protease (MPro)

# 5A: (5RGH) Extension on Z1619978933
$ python3 deepfrag.py --pdb 5rgh --resnum 404 --cname C09

# 5B: (5R81) Extension on Z1367324110
$ python3 deepfrag.py --pdb 5r81 --resnum 1001 --cname C07
```

## Overview

- `config`: fixed configuration information (e.g., TRAIN/VAL/TEST partitions)
- `configurations`: benchmark model configurations (see
  [`configurations/README.md`](configurations/README.md))
- `data`: training/inference data (see [`data/README.md`](data/README.md))
- `leadopt`: main module code
  - `models`: pytorch architecture definitions
  - `data_util.py`: utility code for reading packed fragment/fingerprint data
      files
  - `grid_util.py`: GPU-accelerated grid generation code
  - `metrics.py`: pytorch implementations of several metrics
  - `model_conf.py`: contains code to configure and train models
  - `util.py`: utility code for rdkit/openbabel processing
- `scripts`: data processing scripts (see
  [`scripts/README.md`](scripts/README.md))
- `train.py`: CLI interface to launch training runs

## Dependencies

You can build a virtualenv with the requirements:

```sh
$ python3 -m venv leadopt_env
$ source ./leadopt_env/bin/activate
$ pip install -r requirements.txt
$ pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

<!-- pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -->
<!-- 2.1.2+cu118 -->

Note: `Cuda 10.1` is required during training

## Training

To train a model, you can use the `train.py` utility script. You can specify
model parameters as command line arguments or load parameters from a
configuration args.json file.

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

`save_path` is a directory to save the best model. The directory will be
created if it doesn't exist. If this is not provided, the model will not be
saved.

`wandb_project` is an optional wandb project name. If provided, the run will
be logged to wandb.

See below for available models and model-specific parameters:

## Leadopt Models

In this repository, trainable models are subclasses of
`model_conf.LeadoptModel`. This class encapsulates model configuration
arguments and pytorch models and enables saving and loading multi-component
models.

```py
from leadopt.model_conf import LeadoptModel, MODELS

model = MODELS['voxel']({args...})
model.train(save_path='./mymodel')

...

model2 = LeadoptModel.load('./mymodel')
```

Internally, model arguments are configured by setting up an `argparse` parser
and passing around a `dict` of configuration parameters in `self._args`.

### VoxelNet

```text
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
