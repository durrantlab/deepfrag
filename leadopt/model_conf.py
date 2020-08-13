
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm

from leadopt.models.voxel import VoxelFingerprintNet
from leadopt.data_util import FragmentDataset, FingerprintDataset, LIG_TYPER,\
    REC_TYPER
from leadopt.grid_util import get_batch
from leadopt.metrics import mse, bce, tanimoto, cos, top_k_acc,\
    average_support, inside_support

from config import partitions


def _do_mean(fn):
    def g(yp, yt):
        return torch.mean(fn(yp,yt))
    return g


def _direct_loss(fingerprints, fn):
    return _do_mean(fn)


DIST_FN = {
    'mse': mse,
    'bce': bce,
    'cos': cos,
    'tanimoto': tanimoto
}


LOSS_TYPE = {
    # minimize distance to target fingeprint
    'direct': _direct_loss,

    # minimize distance to target and maximize distance to all other
    'support_v1': average_support,

    # support, limited to closer points
    'support_v2': average_support,
}


class RunLog(object):
    def __init__(self, args, models, wandb_project=None):
        """Initialize a run logger.
        
        Args:
            args: command line training arguments
            models: {name: model} mapping
            wandb_project: a project to initialize wandb or None
        """
        self._use_wandb = wandb_project != None
        if self._use_wandb:
            wandb.init(
                project=wandb_project,
                config=args
            )
    
            for m in models:
                wandb.watch(models[m])

    def log(self, x):
        if self._use_wandb:
            wandb.log(x)


class MetricTracker(object):
    def __init__(self, name, metric_fns):
        self._name = name
        self._metric_fns = metric_fns
        self._metrics = {}

    def evaluate(self, yp, yt):
        for m in self._metric_fns:
            self.update(m, self._metric_fns[m](yp, yt))

    def update(self, name, metric):
        if type(metric) is dict:
            for subname in metric:
                fullname = '%s_%s' % (self._name, subname)
                if not fullname in self._metrics:
                    self._metrics[fullname] = 0
                self._metrics[fullname] += metric[subname]
        else:
            fullname = '%s_%s' % (self._name, name)
            if not fullname in self._metrics:
                self._metrics[fullname] = 0
            self._metrics[fullname] += metric

    def normalize(self, size):
        for m in self._metrics:
            self._metrics[m] /= size

    def clear(self):
        self._metrics = {}

    def get(self, name):
        fullname = '%s_%s' % (self._name, name)
        return self._metrics[fullname]

    def get_all(self):
        return self._metrics


class LeadoptModel(object):
    """Abstract LeadoptModel base class."""

    @staticmethod
    def setup_base_parser(parser):
        """Configures base parser arguments."""
        parser.add_argument('--wandb_project', default=None, help='''
        Set this argument to track run in wandb.
        ''')

    @staticmethod
    def setup_parser(sub):
        """Adds arguments to a subparser.
        
        Args:
            sub: an argparse subparser
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, path):
        """Load model configuration saved with save().

        Call LeadoptModel.load to infer model type.
        Or call subclass.load to load a specific model type.
        
        Args:
            path: full path to saved model
        """
        args_path = os.path.join(path, 'args.json')
        args = json.loads(open(args_path, 'r').read())

        model_type = MODELS[args['version']] if cls is LeadoptModel else cls

        instance = model_type(args, with_log=False)
        for name in instance._models:
            model_path = os.path.join(path, '%s.pt' % name)
            instance._models[name].load_state_dict(torch.load(model_path))

        return instance

    def __init__(self, args, with_log=True):
        self._args = args
        self._models = self.init_models()
        if with_log:
            wandb_project = None
            if 'wandb_project' in self._args:
                wandb_project = self._args['wandb_project']

            self._log = RunLog(self._args, self._models, wandb_project)

    def save(self, path):
        """Save model configuration to a path.
        
        Args:
            path: path to an existing directory to save models
        """
        os.makedirs(path, exist_ok=True)

        args_path = os.path.join(path, 'args.json')
        open(args_path, 'w').write(json.dumps(self._args))

        for name in self._models:
            model_path = os.path.join(path, '%s.pt' % name)
            torch.save(self._models[name].state_dict(), model_path)

    def init_models(self):
        """Initializes any pytorch models.

        Returns a dict of name->model mapping:
        {'model_1': m1, 'model_2': m2, ...}
        """
        return {}

    def train(self, save_path=None):
        """Train the models."""
        raise NotImplementedError()


class VoxelNet(LeadoptModel):
    @staticmethod
    def setup_parser(sub):
        # testing
        sub.add_argument('--no_partitions', action='store_true', default=False, help='''
        If set, disable the use of TRAIN/VAL partitions during training.
        ''')

        # dataset
        sub.add_argument('-f', '--fragments', required=True, help='''
        Path to fragments file.
        ''')
        sub.add_argument('-fp', '--fingerprints', required=True, help='''
        Path to fingerprints file.
        ''')

        # training parameters
        sub.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
        sub.add_argument('--num_epochs', type=int, default=50, help='''
        Number of epochs to train for.
        ''')
        sub.add_argument('--test_steps', type=int, default=500, help='''
        Number of evaluation steps per epoch.
        ''')
        sub.add_argument('-b', '--batch_size', default=32, type=int)

        # grid generation
        sub.add_argument('--grid_width', type=int, default=24)
        sub.add_argument('--grid_res', type=float, default=1)

        # fragment filtering
        sub.add_argument('--fdist_min', type=float, help='''
        Ignore fragments closer to the receptor than this distance (Angstroms).
        ''')
        sub.add_argument('--fdist_max', type=float, help='''
        Ignore fragments further from the receptor than this distance (Angstroms).
        ''')
        sub.add_argument('--fmass_min', type=float, help='''
        Ignore fragments smaller than this mass (Daltons).
        ''')
        sub.add_argument('--fmass_max', type=float, help='''
        Ignore fragments larger than this mass (Daltons).
        ''')

        # receptor/parent options
        sub.add_argument('--ignore_receptor', action='store_true', default=False)
        sub.add_argument('--ignore_parent', action='store_true', default=False)
        sub.add_argument('-rec_typer', required=True, choices=[k for k in REC_TYPER])
        sub.add_argument('-lig_typer', required=True, choices=[k for k in LIG_TYPER])
        sub.add_argument('-rec_channels', required=True, type=int)
        sub.add_argument('-lig_channels', required=True, type=int)

        # model parameters
        sub.add_argument('--in_channels', type=int, default=18)
        sub.add_argument('--output_size', type=int, default=2048)
        sub.add_argument('--pad', default=False, action='store_true')
        sub.add_argument('--blocks', nargs='+', type=int, default=[32,64])
        sub.add_argument('--fc', nargs='+', type=int, default=[2048])
        sub.add_argument('--use_all_labels', default=False, action='store_true')
        sub.add_argument('--dist_fn', default='mse', choices=[k for k in DIST_FN])
        sub.add_argument('--loss', default='direct', choices=[k for k in LOSS_TYPE])

    def init_models(self):
        voxel = VoxelFingerprintNet(
            in_channels=self._args['in_channels'],
            output_size=self._args['output_size'],
            blocks=self._args['blocks'],
            fc=self._args['fc'],
            pad=self._args['pad']
        ).cuda()
        return {'voxel': voxel}

    def train(self, save_path=None):

        print('[*] Loading data...', flush=True)
        train_dat = FragmentDataset(
            self._args['fragments'],
            rec_typer=REC_TYPER[self._args['rec_typer']],
            lig_typer=LIG_TYPER[self._args['lig_typer']],
            filter_rec=(
                partitions.TRAIN if not self._args['no_partitions'] else None),
            fdist_min=self._args['fdist_min'], 
            fdist_max=self._args['fdist_max'], 
            fmass_min=self._args['fmass_min'], 
            fmass_max=self._args['fmass_max'],
            verbose=True
        )

        val_dat = FragmentDataset(
            self._args['fragments'],
            rec_typer=REC_TYPER[self._args['rec_typer']],
            lig_typer=LIG_TYPER[self._args['lig_typer']],
            filter_rec=(
                partitions.VAL if not self._args['no_partitions'] else None),
            fdist_min=self._args['fdist_min'], 
            fdist_max=self._args['fdist_max'], 
            fmass_min=self._args['fmass_min'], 
            fmass_max=self._args['fmass_max'],
            verbose=True
        )

        fingerprints = FingerprintDataset(self._args['fingerprints'])

        train_smiles = train_dat.get_valid_smiles()
        val_smiles = val_dat.get_valid_smiles()
        all_smiles = list(set(train_smiles) | set(val_smiles))

        train_fingerprints = fingerprints.for_smiles(train_smiles).cuda()
        val_fingerprints = fingerprints.for_smiles(val_smiles).cuda()
        all_fingerprints = fingerprints.for_smiles(all_smiles).cuda()

        # fingerprint metrics
        print('[*] Train smiles: %d' % len(train_smiles))
        print('[*] Val smiles: %d' % len(val_smiles))
        print('[*] All smiles: %d' % len(all_smiles))

        print('[*] Train smiles: %d' % train_fingerprints.shape[0])
        print('[*] Val smiles: %d' % val_fingerprints.shape[0])
        print('[*] All smiles: %d' % all_fingerprints.shape[0])

        print('[*] Training...', flush=True)
        opt = torch.optim.Adam(
            self._models['voxel'].parameters(), lr=self._args['learning_rate'])
        steps_per_epoch = len(train_dat) // self._args['batch_size']
        
        # configure metrics
        dist_fn = DIST_FN[self._args['dist_fn']]

        loss_fingerprints = train_fingerprints
        if self._args['use_all_labels']:
            loss_fingerprints = all_fingerprints

        loss_fn = LOSS_TYPE[self._args['loss']](loss_fingerprints, dist_fn)

        train_metrics = MetricTracker('train', {
            'all': top_k_acc(all_fingerprints, dist_fn, [1,5,10,50,100], pre='all')
        })
        val_metrics = MetricTracker('val', {
            'all': top_k_acc(all_fingerprints, dist_fn, [1,5,10,50,100], pre='all'),
            'val': top_k_acc(val_fingerprints, dist_fn, [1,5,10,50,100], pre='val'),
        })

        best_loss = None

        for epoch in range(self._args['num_epochs']):
            self._models['voxel'].train()
            train_pbar = tqdm.tqdm(
                range(steps_per_epoch),
                desc='Train (epoch %d)' % epoch
            )
            for step in train_pbar:
                torch_grid, examples = get_batch(
                    train_dat,
                    self._args['rec_channels'],
                    self._args['lig_channels'], 
                    batch_size=self._args['batch_size'],
                    batch_set=None,
                    width=self._args['grid_width'],
                    res=self._args['grid_res'],
                    ignore_receptor=self._args['ignore_receptor'],
                    ignore_parent=self._args['ignore_parent']
                )

                smiles = [example['smiles'] for example in examples]
                correct_fp = torch.Tensor(
                    fingerprints.for_smiles(smiles)).cuda()

                predicted_fp = self._models['voxel'](torch_grid)

                loss = loss_fn(predicted_fp, correct_fp)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_metrics.update('loss', loss)
                train_metrics.evaluate(predicted_fp, correct_fp)
                self._log.log(train_metrics.get_all())
                train_metrics.clear()

            self._models['voxel'].eval()

            val_pbar = tqdm.tqdm(
                range(self._args['test_steps']),
                desc='Val %d' % epoch
            )
            with torch.no_grad():
                for step in val_pbar:
                    torch_grid, examples = get_batch(
                        val_dat,
                        self._args['rec_channels'],
                        self._args['lig_channels'], 
                        batch_size=self._args['batch_size'],
                        batch_set=None,
                        width=self._args['grid_width'],
                        res=self._args['grid_res'],
                        ignore_receptor=self._args['ignore_receptor'],
                        ignore_parent=self._args['ignore_parent']
                    )

                    smiles = [example['smiles'] for example in examples]
                    correct_fp = torch.Tensor(
                        fingerprints.for_smiles(smiles)).cuda()

                    predicted_fp = self._models['voxel'](torch_grid)

                    loss = loss_fn(predicted_fp, correct_fp)

                    val_metrics.update('loss', loss)
                    val_metrics.evaluate(predicted_fp, correct_fp)

                val_metrics.normalize(self._args['test_steps'])
                self._log.log(val_metrics.get_all())

                val_loss = val_metrics.get('loss')
                if best_loss is None or val_loss < best_loss:
                    # save new best model
                    best_loss = val_loss
                    print('[*] New best loss: %f' % best_loss, flush=True)
                    if save_path:
                        self.save(save_path)

                val_metrics.clear()


MODELS = {
    'voxelnet': VoxelNet
}
