
import argparse
import os
import random


def load_grouped(index):
    dat = open(index, 'r').read().strip().split('\n')[6:]

    # uniprot: PDB
    targets_by_uniprot = {}

    # group by uniprot id
    for line in dat:
        sp = line.split()
        if not sp[2] in targets_by_uniprot:
            targets_by_uniprot[sp[2]] = []
        targets_by_uniprot[sp[2]].append(sp[0])

    return targets_by_uniprot


def split_groups(grp, ratio):
    '''split into two groups'''    
    # randomly shuffle uniprot IDs
    uniprot = sorted([x for x in grp])
    random.shuffle(uniprot)
    
    # count ids
    total = sum(len(grp[x]) for x in grp)
    
    a = {}
    b = {}
  
    idx = 0
    acc = 0
    for idx in range(len(uniprot)):
        a[uniprot[idx]] = grp[uniprot[idx]]
        acc += len(grp[uniprot[idx]])
        
        if acc > (total * ratio):
            break
            
    for idx in range(idx+1, len(uniprot)):
        b[uniprot[idx]] = grp[uniprot[idx]]
            
    return a, b


def flatten(grp):
    f = []
    for g in grp:
        f += grp[g]
    return f


def make_split(index, seed):
    targets_by_uniprot = load_grouped(index)
    random.seed(seed)

    print(f'# PDBBind split generated with seed: {seed}\n')

    # 80/20 TRAIN/TEST
    comb, test = split_groups(targets_by_uniprot, 0.8)
    f_test = flatten(test)

    print(f'# test: {len(f_test)}')

    # 80/20 TRAIN/VAL (within train set)
    train, val = split_groups(comb, 0.8)

    f_val = flatten(val)
    f_train = flatten(train)

    print(f'# val: {len(f_val)}')
    print(f'# train: {len(f_train)}')

    print('\n')

    print('TEST =', sorted(f_test), '\n')
    print('VAL =', sorted(f_val), '\n')
    print('TRAIN =', sorted(f_train), '\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('index', help='Path to INDEX_general_PL_name')
    parser.add_argument('-s', '--seed', default=0, help='Seed for split randomization')

    args = parser.parse_args()

    make_split(args.index, args.seed)


if __name__=='__main__':
    main()
