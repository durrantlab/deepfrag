# Copyright 2021 Jacob Durrant

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


import argparse
import os

import numpy as np

from moad_util import parse_moad


def all_smi(families):
    smi = []
    for f in families:
        for t in f.targets:
            for lig in t.ligands:
                smi.append(lig[1])
    return set(smi)

def do_split(families, pa=0.6):
    fa = []
    fb = []

    for f in families:
        if np.random.rand() < pa:
            fa.append(f)
        else:
            fb.append(f)

    return (fa, fb)

def split_smi(smi):
    l = list(smi)
    sz = len(l)

    np.random.shuffle(l)

    return (set(l[:sz//2]), set(l[sz//2:]))

def split_smi3(smi):
    l = list(smi)
    sz = len(l)

    np.random.shuffle(l)

    v = sz//3
    return (set(l[:v]), set(l[v:v*2]), set(l[v*2:]))

def get_ids(fam):
    ids = []
    for f in fam:
        for t in f.targets:
            ids.append(t.pdb_id)
    return ids

def gen_split(csv, output, seed=7):
    np.random.seed(seed)

    moad_families, moad_targets = parse_moad(csv)

    train, other = do_split(moad_families, pa=0.6)
    val, test = do_split(other, pa=0.5)

    train_sum = np.sum([len(x.targets) for x in train])
    val_sum = np.sum([len(x.targets) for x in val])
    test_sum = np.sum([len(x.targets) for x in test])

    print('[Targets] (Train: %d) (Val: %d) (Test %d)' % (train_sum, val_sum, test_sum))

    train_smi = all_smi(train)
    val_smi = all_smi(val)
    test_smi = all_smi(test)

    train_smi_uniq = train_smi - (val_smi | test_smi)
    val_smi_uniq = val_smi - (train_smi | test_smi)
    test_smi_uniq = test_smi - (val_smi | train_smi)

    print('[Unique ligands] (Train: %d) (Val: %d) (Test %d)' % (
        len(train_smi_uniq), len(val_smi_uniq), len(test_smi_uniq)))

    print('[Total unique ligands] %d' % len(train_smi | val_smi | test_smi))

    split_train_val = (train_smi & val_smi) - test_smi
    split_train_test = (train_smi & test_smi) - val_smi
    split_val_test = (val_smi & test_smi) - train_smi

    split_all = (train_smi & val_smi & test_smi)

    split_train_val_a, split_train_val_b = split_smi(split_train_val)
    split_train_test_a, split_train_test_b = split_smi(split_train_test)
    split_val_test_a, split_val_test_b = split_smi(split_val_test)

    split_all_train, split_all_val, split_all_test = split_smi3(split_all)

    train_full = (train_smi_uniq | split_train_val_a | split_train_test_a | split_all_train)
    val_full = (val_smi_uniq | split_train_val_b | split_val_test_a | split_all_val)
    test_full = (test_smi_uniq | split_train_test_b | split_val_test_b | split_all_test)

    print('[Full ligands] (Train: %d) (Val: %d) (Test %d)' % (
        len(train_full), len(val_full), len(test_full)))

    mixed = (train_full & val_full) | (val_full & test_full) | (train_full & test_full)

    train_ids = sorted(get_ids(train))
    val_ids = sorted(get_ids(val))
    test_ids = sorted(get_ids(test))

    train_s = sorted(train_full)
    val_s = sorted(val_full)
    test_s = sorted(test_full)

    # Format as a python file.
    out = ''
    out += 'TRAIN = ' + repr(train_ids).replace(' ','') + '\n'
    out += 'TRAIN_SMI = ' + repr(train_s).replace(' ','') + '\n'
    out += 'VAL = ' + repr(val_ids).replace(' ','') + '\n'
    out += 'VAL_SMI = ' + repr(val_s).replace(' ','') + '\n'
    out += 'TEST = ' + repr(test_ids).replace(' ','') + '\n'
    out += 'TEST_SMI = ' + repr(test_s).replace(' ','') + '\n'

    open(output, 'w').write(out)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv', required=True, help='Path to every.csv file')
    parser.add_argument('-s', '--seed', required=False, default=7, type=int, help='Integer seed')
    parser.add_argument('-o', '--output', default='moad_partitions.py', help='Output file path (.py)')

    args = parser.parse_args()

    gen_split(args.csv, args.output, args.seed)
    print('Done!')

if __name__=='__main__':
    main()
