
import os
import sys
import argparse

import torch
import torch.nn as nn
import wandb
import tqdm
import numpy as np

from leadopt.data_util import FragmentDataset, REC_TYPER, LIG_TYPER

from config import partitions

from leadopt.model_conf import MODELS


def main():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-s', '--save', action='store_true', default=False)

    # dataset
    parser.add_argument('-f', '--fragments', required=True)
    parser.add_argument('-fp', '--fingerprints', required=True)

    # fingerprints
    parser.add_argument('-n', '--normalize_fingerprint', action='store_true', default=False)

    # training parameters
    parser.add_argument('-lr','--learning_rate',type=float,default=1e-4)
    parser.add_argument('--num_epochs',type=int,default=50)
    parser.add_argument('--test_steps',type=int,default=500)
    parser.add_argument('-b', '--batch_size',default=32,type=int)
    parser.add_argument('-w', '--weighted',default=False,action='store_true') # weight by class freq

    # grid generration
    parser.add_argument('--grid_width',type=int,default=24)
    parser.add_argument('--grid_res',type=float,default=1)

    # fragment filtering
    parser.add_argument('--fdist_min',type=float)
    parser.add_argument('--fdist_max',type=float)
    parser.add_argument('--fmass_min',type=float)
    parser.add_argument('--fmass_max',type=float)

    # receptor/parent options
    parser.add_argument('--ignore_receptor',action='store_true',default=False)
    parser.add_argument('--ignore_parent',action='store_true',default=False)
    parser.add_argument('-rec_typer', required=True, choices=[k for k in REC_TYPER])
    parser.add_argument('-lig_typer', required=True, choices=[k for k in LIG_TYPER])
    parser.add_argument('-rec_channels', required=True, type=int)
    parser.add_argument('-lig_channels', required=True, type=int)

    subparsers = parser.add_subparsers(dest='version')

    # add models
    for m in MODELS:
        MODELS[m].setup_parser(m, subparsers)

    # parse arguments
    args = parser.parse_args()

    if args.version is None:
        parser.print_help()
        exit(0)

    print(args)
    print(' '.join(sys.argv))

    if not args.save:
        print('[!] Not saving model!')
    save_best = args.save

    print('[*] Init model...',flush=True)
    model = MODELS[args.version].build_model(args)

    # load data
    print('[*] Loading data...',flush=True)
    train_dat = FragmentDataset(
        args.fragments,
        args.fingerprints,
        rec_typer=REC_TYPER[args.rec_typer],
        lig_typer=LIG_TYPER[args.lig_typer],
        filter_rec=partitions.TRAIN, 
        fdist_min=args.fdist_min, 
        fdist_max=args.fdist_max, 
        fmass_min=args.fmass_min, 
        fmass_max=args.fmass_max,
        verbose=True
    )

    test_dat = FragmentDataset(
        args.fragments,
        args.fingerprints,
        rec_typer=REC_TYPER[args.rec_typer],
        lig_typer=LIG_TYPER[args.lig_typer],
        filter_rec=partitions.VAL, 
        fdist_min=args.fdist_min, 
        fdist_max=args.fdist_max, 
        fmass_min=args.fmass_min, 
        fmass_max=args.fmass_max,
        verbose=True
    )

    # normalize by train data valid
    dat = train_dat.fingerprints['fingerprint_data'][train_dat.valid_fingerprints]

    std = np.ones(dat.shape[1])
    mean = np.zeros(dat.shape[1])
    if args.normalize_fingerprint:
        std = np.std(dat, axis=0)
        mean = np.mean(dat, axis=0)

        train_dat.normalize_fingerprints(std, mean)
        test_dat.normalize_fingerprints(std, mean)

    print('[:] Normalized with mean: %s, std: %s' % (repr(mean), repr(std)))

    print('\t[:] Train set: %d' % len(train_dat))
    print('\t[:] Test set: %d' % len(test_dat))
    print('')
    print('\t[:] Train fingerprints: %d' % len(train_dat.valid_fingerprints))
    print('\t[:] Test fingerprints: %d' % len(test_dat.valid_fingerprints))

    # create metrics
    metrics = MODELS[args.version].get_metrics(args, std, mean, train_dat, test_dat)

    # create loss function
    loss_fn = MODELS[args.version].get_loss(args, std, mean, train_dat, test_dat)

    train_func = MODELS[args.version].get_train_mode()

    print('[*] Training...', flush=True)
    train_func(
        model,
        './',
        train_dat,
        test_dat,
        metrics,
        loss_fn,
        args
    )


if __name__=='__main__':
    main()
