
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm
import numpy as np

from leadopt.data_util import FragmentDataset
from leadopt.grid_util import get_batch, get_batch_dual, get_batch_full
from leadopt.metrics import tanimoto

from config import partitions


def train(model, run_path, train_dat, test_dat, metrics, loss_fn, args):
    
    # track run
    config={
        '_model': model.__class__.__name__,
        '_run_path': run_path,
        '_cwd': os.getcwd(),
    }
    config.update(args.__dict__)

    wandb.init(
        project='leadopt_pytorch',
        config=config
    )
    
    wandb.watch(model)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    steps_per_epoch = len(train_dat) // args.batch_size
    
    best_val_loss = None
    
    # train!
    for epoch in range(args.num_epochs):
        
        # switch to train mode
        model.train()
        
        for step in tqdm.tqdm(range(steps_per_epoch), desc='Epoch %d' % epoch):
            # get batch
            w = np.ones(args.batch_size)
            if args.weighted:
                # weight by class frequency
                t, fp, freq, _ = get_batch(
                    train_dat, 
                    batch_size=args.batch_size, 
                    width=args.grid_width, 
                    res=args.grid_res, 
                    ignore_receptor=args.ignore_receptor,
                    ignore_parent=args.ignore_parent,
                    include_freq=True
                )
                w = (1 / freq)
            else:
                t, fp, _ = get_batch(
                    train_dat, 
                    batch_size=args.batch_size, 
                    width=args.grid_width, 
                    res=args.grid_res, 
                    ignore_receptor=args.ignore_receptor,
                    ignore_parent=args.ignore_parent
                )
        
            y = model(t)

            loss = 0

            if args.weighted:
                for i in range(args.batch_size):
                    loss += loss_fn(y[i], fp[i]) * w[i]
                loss /= args.batch_size
            else:
                loss += loss_fn(y,fp)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            calc_metrics = {'loss': loss}
            for m in metrics:
                r = metrics[m](y, fp)

                # dict: multiple values
                if type(r) is dict:
                    calc_metrics.update(r)
                else:
                    calc_metrics[m] = r
            
            wandb.log(calc_metrics)
    
        # switch to eval mode
        model.eval()
        
        val_metrics = {'val_loss': 0}
        for m in metrics:
            val_metrics['val_%s' % m] = 0
            # val_metrics['val_weighted_%s' % m] = 0
        
        with torch.no_grad():
            for step in tqdm.tqdm(range(args.test_steps), desc='Val %d' % epoch):
                # get batch
                w = torch.ones(args.batch_size).cuda()
                if args.weighted:
                    # weight by class frequency
                    t, fp, freq, _ = get_batch(
                        train_dat, 
                        batch_size=args.batch_size, 
                        width=args.grid_width, 
                        res=args.grid_res, 
                        ignore_receptor=args.ignore_receptor,
                        ignore_parent=args.ignore_parent,
                        include_freq=True
                    )
                    w = (1 / freq)
                else:
                    t, fp, _ = get_batch(
                        train_dat, 
                        batch_size=args.batch_size, 
                        width=args.grid_width, 
                        res=args.grid_res, 
                        ignore_receptor=args.ignore_receptor,
                        ignore_parent=args.ignore_parent
                    )
            
                y = model(t)

                loss = 0
                if args.weighted:
                    for i in range(args.batch_size):
                        loss += loss_fn(y[i], fp[i]) * w[i]
                    loss /= args.batch_size
                else:
                    loss += loss_fn(y,fp)
                
                val_metrics['val_loss'] += loss

                for m in metrics:
                    r = metrics[m](y, fp)

                    # dict: multiple values
                    if type(r) is dict:
                        calc_metrics.update({
                            'val_%s' % k: r[k] for k in r
                        })
                    else:
                        calc_metrics[m] = r

                # for m in metrics:
                #     wm = 'val_weighted_%s' % m
                #     t = 0
                #     for i in range(args.batch_size):
                #         t += metrics[m](y[i].unsqueeze(0), fp[i].unsqueeze(0)) * w[i]
                #     t /= args.batch_size
                #     val_metrics[wm] += t
            
        val_metrics = {k:val_metrics[k]/args.test_steps for k in val_metrics}
        wandb.log(val_metrics)
        print(val_metrics)
        
        if best_val_loss is None or val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            if args.save:
                print('[*] Saving...')
                torch.save(model.state_dict(), os.path.join(run_path, 'model_best.pt'))


def train_dual(model, run_path, train_dat, test_dat, metrics, loss_fn, args):

    # track run
    config={
        '_model': model.__class__.__name__,
        '_run_path': run_path,
        '_cwd': os.getcwd(),
    }
    config.update(args.__dict__)

    wandb.init(
        project='leadopt_pytorch',
        config=config
    )
    
    wandb.watch(model)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    steps_per_epoch = len(train_dat) // args.batch_size
    
    best_val_loss = None
    
    # train!
    for epoch in range(args.num_epochs):
        
        # switch to train mode
        model.train()
        
        for step in tqdm.tqdm(range(steps_per_epoch), desc='Epoch %d' % epoch):
            # get batch
            t, fp, yt, _  = get_batch_dual(
                train_dat, 
                batch_size=args.batch_size, 
                width=args.grid_width, 
                res=args.grid_res, 
                ignore_receptor=args.ignore_receptor,
                ignore_parent=args.ignore_parent
            )
        
            yp = model(t, fp)
            loss = loss_fn(yp, yt)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            calc_metrics = {'loss': loss}
            for m in metrics:
                calc_metrics[m] = metrics[m](yp, yt)
            
            wandb.log(calc_metrics)
    
        # switch to eval mode
        model.eval()
        
        val_metrics = {'val_loss': 0}
        for m in metrics:
            val_metrics['val_%s' % m] = 0
        
        with torch.no_grad():
            for step in tqdm.tqdm(range(args.test_steps), desc='Val %d' % epoch):
                # get batch
                t, fp, yt, _  = get_batch_dual(
                    train_dat, 
                    batch_size=args.batch_size, 
                    width=args.grid_width, 
                    res=args.grid_res, 
                    ignore_receptor=args.ignore_receptor,
                    ignore_parent=args.ignore_parent
                )
            
                yp = model(t, fp)
                loss = loss_fn(yp, yt)
                
                val_metrics['val_loss'] += loss

                for m in metrics:
                    val_metrics['val_%s' % m] += metrics[m](yp, yt)
            
        val_metrics = {k:val_metrics[k]/args.test_steps for k in val_metrics}
        wandb.log(val_metrics)
        print(val_metrics)
        
        if best_val_loss is None or val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            if args.save:
                print('[*] Saving...')
                torch.save(model.state_dict(), os.path.join(run_path, 'model_best.pt'))


def train_latent(model, run_path, train_dat, test_dat, metrics, loss_fn, args):

    # track run
    config={
        '_run_path': run_path,
        '_cwd': os.getcwd(),
    }
    config.update(args.__dict__)

    wandb.init(
        project='leadopt_pytorch',
        config=config
    )

    encoder = model[0]
    decoder = model[1]
    context_encoder = model[2]
    
    wandb.watch(encoder)
    wandb.watch(decoder)
    wandb.watch(context_encoder)
    
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(context_encoder.parameters()), 
        lr=args.learning_rate)
    
    steps_per_epoch = len(train_dat) // args.batch_size
    
    best_val_loss = None

    LOSSES = {
        'bce': nn.BCELoss(),
        'mse': F.mse_loss,
        'none': lambda a,b:0
    }

    frag_loss = LOSSES[args.frag_loss]
    context_loss = LOSSES[args.context_loss]
    
    # train!
    for epoch in range(args.num_epochs):
        
        # switch to train mode
        encoder.train()
        decoder.train()
        context_encoder.train()
        
        for step in tqdm.tqdm(range(steps_per_epoch), desc='Epoch %d' % epoch):
            # get batch
            t_context, t_frag, _  = get_batch_full(
                train_dat, 
                batch_size=args.batch_size, 
                width=args.grid_width, 
                res=args.grid_res, 
                ignore_receptor=args.ignore_receptor,
                ignore_parent=args.ignore_parent
            )

            t_frag = torch.clamp(t_frag, 0, 1)
        
            # reconstruction loss
            z = encoder(t_frag)
            reconstructed = decoder(z)
            reconstruction_loss = frag_loss(reconstructed, t_frag)

            # target loss
            z2 = context_encoder(t_context)
            target_loss = context_loss(z2, z.detach())

            loss = reconstruction_loss + target_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            calc_metrics = {
                'loss': loss,
                'reconstruction_loss': reconstruction_loss,
                'target_loss': target_loss
            }
            for m in metrics:
                calc_metrics[m] = metrics[m](reconstructed, t_frag, z, z2)
            
            wandb.log(calc_metrics)
    
        # switch to eval mode
        encoder.eval()
        decoder.eval()
        context_encoder.eval()
        
        val_metrics = {
            'val_loss': 0,
            'val_reconstruction_loss': 0,
            'val_target_loss': 0
        }
        for m in metrics:
            val_metrics['val_%s' % m] = 0
        
        with torch.no_grad():
            for step in tqdm.tqdm(range(args.test_steps), desc='Val %d' % epoch):
                # get batch
                t_context, t_frag, _  = get_batch_full(
                    test_dat, 
                    batch_size=args.batch_size, 
                    width=args.grid_width, 
                    res=args.grid_res, 
                    ignore_receptor=args.ignore_receptor,
                    ignore_parent=args.ignore_parent
                )

                t_frag = torch.clamp(t_frag, 0, 1)
            
                # reconstruction loss
                z = encoder(t_frag)
                reconstructed = decoder(z)
                reconstruction_loss = frag_loss(reconstructed, t_frag)

                # target loss
                z2 = context_encoder(t_context)
                target_loss = context_loss(z2, z.detach())

                loss = reconstruction_loss + target_loss
                
                val_metrics['val_loss'] += loss
                val_metrics['val_reconstruction_loss'] += reconstruction_loss
                val_metrics['val_target_loss'] += target_loss

                for m in metrics:
                    val_metrics['val_%s' % m] += metrics[m](reconstructed, t_frag, z, z2)
            
        val_metrics = {k:val_metrics[k]/args.test_steps for k in val_metrics}
        wandb.log(val_metrics)
        print(val_metrics)
        
        if best_val_loss is None or val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            if args.save:
                print('[*] Saving...')
                torch.save(encoder.state_dict(), os.path.join(run_path, 'encoder_best.pt'))
                torch.save(decoder.state_dict(), os.path.join(run_path, 'decoder_best.pt'))
                torch.save(context_encoder.state_dict(), os.path.join(run_path, 'context_encoder_best.pt'))


def train_latent_direct(model, run_path, train_dat, test_dat, metrics, loss_fn, args):

    # track run
    config={
        '_run_path': run_path,
        '_cwd': os.getcwd(),
    }
    config.update(args.__dict__)

    wandb.init(
        project='leadopt_pytorch',
        config=config
    )

    wandb.watch(model)
    
    opt = torch.optim.Adam(
        list(model.parameters()), 
        lr=args.learning_rate)
    
    steps_per_epoch = len(train_dat) // args.batch_size
    
    best_val_loss = None

    LOSSES = {
        'bce': nn.BCELoss(),
        'mse': F.mse_loss,
        'none': lambda a,b:0
    }

    frag_loss = LOSSES[args.frag_loss]
    
    # train!
    for epoch in range(args.num_epochs):
        
        # switch to train mode
        model.train()
        
        for step in tqdm.tqdm(range(steps_per_epoch), desc='Epoch %d' % epoch):
            # get batch
            t_context, t_frag, _  = get_batch_full(
                train_dat, 
                batch_size=args.batch_size, 
                width=args.grid_width, 
                res=args.grid_res, 
                ignore_receptor=args.ignore_receptor,
                ignore_parent=args.ignore_parent
            )

            # reconstruction loss
            out = model(t_context)
            loss = frag_loss(out, t_frag)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            calc_metrics = {
                'loss': loss
            }
            for m in metrics:
                calc_metrics[m] = metrics[m](out, t_frag)
            
            wandb.log(calc_metrics)
    
        # switch to eval mode
        model.eval()
        
        val_metrics = {
            'val_loss': 0
        }
        for m in metrics:
            val_metrics['val_%s' % m] = 0
        
        with torch.no_grad():
            for step in tqdm.tqdm(range(args.test_steps), desc='Val %d' % epoch):
                # get batch
                t_context, t_frag, _  = get_batch_full(
                    test_dat, 
                    batch_size=args.batch_size, 
                    width=args.grid_width, 
                    res=args.grid_res, 
                    ignore_receptor=args.ignore_receptor,
                    ignore_parent=args.ignore_parent
                )

                out = model(t_context)
                loss = frag_loss(out, t_frag)
                
                val_metrics['val_loss'] += loss

                for m in metrics:
                    val_metrics['val_%s' % m] += metrics[m](out, t_frag)
            
        val_metrics = {k:val_metrics[k]/args.test_steps for k in val_metrics}
        wandb.log(val_metrics)
        print(val_metrics)
        
        if best_val_loss is None or val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            if args.save:
                print('[*] Saving...')
                torch.save(model.state_dict(), os.path.join(run_path, 'model.pt'))


def train_attention(model, run_path, train_dat, test_dat, metrics, loss_fn, args):
    
    # track run
    config={
        '_model': model.__class__.__name__,
        '_run_path': run_path,
        '_cwd': os.getcwd(),
    }
    config.update(args.__dict__)

    wandb.init(
        project='leadopt_pytorch',
        config=config
    )
    
    wandb.watch(model)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    steps_per_epoch = len(train_dat) // args.batch_size
    
    best_val_loss = None
    
    # train!
    for epoch in range(args.num_epochs):
        
        # switch to train mode
        model.train()
        
        for step in tqdm.tqdm(range(steps_per_epoch), desc='Epoch %d' % epoch):
            # get batch
            w = np.ones(args.batch_size)
            if args.weighted:
                # weight by class frequency
                t, fp, freq, _ = get_batch(
                    train_dat, 
                    batch_size=args.batch_size, 
                    width=args.grid_width, 
                    res=args.grid_res, 
                    ignore_receptor=args.ignore_receptor,
                    ignore_parent=args.ignore_parent,
                    include_freq=True
                )
                w = (1 / freq)
            else:
                t, fp, _ = get_batch(
                    train_dat, 
                    batch_size=args.batch_size, 
                    width=args.grid_width, 
                    res=args.grid_res, 
                    ignore_receptor=args.ignore_receptor,
                    ignore_parent=args.ignore_parent
                )
        
            y, att = model(t)

            loss = 0

            if args.weighted:
                for i in range(args.batch_size):
                    loss += loss_fn(y[i], fp[i], att[i]) * w[i]
                loss /= args.batch_size
            else:
                loss += loss_fn(y,fp,att)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            calc_metrics = {'loss': loss}
            for m in metrics:
                calc_metrics[m] = metrics[m](y, fp, att)
            # for m in metrics:
            #     wm = 'weighted_%s' % m
            #     calc_metrics[wm] = 0
            #     for i in range(args.batch_size):
            #         calc_metrics[wm] += metrics[m](y[i].unsqueeze(0), fp[i].unsqueeze(0)) * w[i]
            #     calc_metrics[wm] /= args.batch_size
            
            wandb.log(calc_metrics)
    
        # switch to eval mode
        model.eval()
        
        val_metrics = {'val_loss': 0}
        for m in metrics:
            val_metrics['val_%s' % m] = 0
            # val_metrics['val_weighted_%s' % m] = 0
        
        with torch.no_grad():
            for step in tqdm.tqdm(range(args.test_steps), desc='Val %d' % epoch):
                # get batch
                w = torch.ones(args.batch_size).cuda()
                if args.weighted:
                    # weight by class frequency
                    t, fp, freq, _ = get_batch(
                        train_dat, 
                        batch_size=args.batch_size, 
                        width=args.grid_width, 
                        res=args.grid_res, 
                        ignore_receptor=args.ignore_receptor,
                        ignore_parent=args.ignore_parent,
                        include_freq=True
                    )
                    w = (1 / freq)
                else:
                    t, fp, _ = get_batch(
                        train_dat, 
                        batch_size=args.batch_size, 
                        width=args.grid_width, 
                        res=args.grid_res, 
                        ignore_receptor=args.ignore_receptor,
                        ignore_parent=args.ignore_parent
                    )
            
                y, att = model(t)

                loss = 0
                if args.weighted:
                    for i in range(args.batch_size):
                        loss += loss_fn(y[i], fp[i], att[i]) * w[i]
                    loss /= args.batch_size
                else:
                    loss += loss_fn(y,fp,att)
                
                val_metrics['val_loss'] += loss

                for m in metrics:
                    val_metrics['val_%s' % m] += metrics[m](y, fp, att)
                # for m in metrics:
                #     wm = 'val_weighted_%s' % m
                #     t = 0
                #     for i in range(args.batch_size):
                #         t += metrics[m](y[i].unsqueeze(0), fp[i].unsqueeze(0)) * w[i]
                #     t /= args.batch_size
                #     val_metrics[wm] += t
            
        val_metrics = {k:val_metrics[k]/args.test_steps for k in val_metrics}
        wandb.log(val_metrics)
        print(val_metrics)
        
        if best_val_loss is None or val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            if args.save:
                print('[*] Saving...')
                torch.save(model.state_dict(), os.path.join(run_path, 'model_best.pt'))
