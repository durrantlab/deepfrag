
# MOAD Data

This readme describes how to process raw MOAD data for use in training the DeepFrag model. Note that we already provide fully processed datasets for
training so this section mostly serves as a description of how to process future versions of the MOAD dataset or as an example for researchers looking
to accomplish similar things.

See [`data/README.md`](../data/README.md) for instructions on how to download the packed `moad.h5` data.

See [`config/moad_partitions.py`](../config/moad_partitions.py) for a pre-computed MOAD TRAIN/VAL/TEST split (generated with seed 7).

# 1. Download MOAD datasets

Go to https://bindingmoad.org/Home/download and download `every_part_a.zip` and `every_part_b.zip` and "Binding data" (`every.csv`).

This readme assumes these files are stored in `$MOAD_DIR`.

# 2. Unpack MOAD datasets

Run:

```sh
$ unzip every_part_a.zip
...
$ unzip every_part_b.zip
...
```

# 3. Process MOAD pdb files

The MOAD dataset contains ligand/receptor structures combined in a single pdb file (named with a `.bio<x>` extension). In this step, we will separate the receptor and each ligand into individual files.

Run:

```sh
$ cd $MOAD_DIR && mkdir split
$ python3 scripts/split_moad.py \
    -d $MOAD_DIR/BindingMOAD_2020 \
    -c $MOAD_DIR/every.csv \
    -o $MOAD_DIR/split \
    -n <number of cores>
```

# 4. Generate packed data files.

For training purposes, we pack all of the relevant information into an h5 file so we can load it entirely in memory during training.

This step will produce several similar `.h5` files that can be combined later.

Run:

```sh
$ cd $MOAD_DIR && mkdir packed
$ python3 scripts/process_moad.py \
    -d $MOAD_DIR/split \
    -c $MOAD_DIR/every.csv \
    -o $MOAD_DIR/packed/moad.h5 \
    -n <number of cores> \
    -s <examples per file (e.g. 500)>
```

# 5. Merge packed data files.

```sh
$ python3 scripts/merge_moad.py \
    -i $MOAD_DIR/packed \
    -o moad.h5
```

# 6. Generate MOAD Training splits

```sh
$ python3 scripts/moad_training_splits.py \
    -c $MOAD_DIR/every.csv \
    -s 7 \
    -o moad_partitions.py
```
