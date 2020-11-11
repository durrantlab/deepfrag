
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
