
import torch
import torch.nn as nn
import torch.nn.functional as F

from leadopt.train import train, train_dual, train_latent, train_latent_direct, train_attention
from leadopt.metrics import tanimoto

from leadopt.models.voxel import VoxelFingerprintNet, VoxelFingerprintNet2, VoxelFingerprintNet2b, VoxelFingerprintNet3, VoxelFingerprintNet4
from leadopt.models.latent import LatentEncoder, LatentDecoder
from leadopt.models.encoder_skip import FullSkipV1


def denorm(f, std, mean):
    t_std = torch.Tensor(std).cuda()
    t_mean = torch.Tensor(mean).cuda()

    def g(yp, yt):
        yp = (yp * t_std) + t_mean
        yt = (yt * t_std) + t_mean

        return f(yp, yt)

    return g

def within_k_mass(k):
    def g(yp, yt):
        return torch.mean((torch.abs(yp[:,0] - yt[:,0]) < k).to(torch.float))

    return g


def average_position(fingerprints, fn, norm=True):
    t_fingerprints = torch.Tensor(fingerprints).cuda()

    def fn_br(yp,yt):
        yp_b, yt_b = torch.broadcast_tensors(yp, yt)
        return fn(yp_b.detach(), yt_b.detach())

    def g(yp, yt):
        # correct distance
        p_dist = fn_br(yp, yt)

        c = torch.empty(yp.shape[0])
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = fn_br(yp[i].unsqueeze(0), t_fingerprints)

            # number of fragment that are closer or equal
            count = torch.sum((dist <= p_dist[i]).to(torch.float))

            if norm:
                count /= t_fingerprints.shape[0]

            c[i] = count
            
        score = torch.mean(c)

        return score

    return g


def average_position_mse(fingerprints, norm):
    def fn(yp, yt):
        return torch.sum((yp - yt) ** 2, axis=1)
    return average_position(fingerprints, fn, norm)


def average_position_cos(fingerprints, norm):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def fn(yp, yt):
        return 1 - cos(yp,yt)
    return average_position(fingerprints, fn, norm)


def average_position_tanimoto(fingerprints, norm):
    def tanimoto(yp, yt):
        intersect = torch.sum(yt * torch.round(yp), axis=1)
        union = torch.sum(torch.clamp(yt + torch.round(yp), 0, 1), axis=1)
        return 1 - (intersect / union)
    return average_position(fingerprints, tanimoto, norm)


def average_position_bce(fingerprints, norm):
    def fn(yp, yt):
        return torch.sum(F.binary_cross_entropy(yp,yt,reduction='none'), axis=1)
    return average_position(fingerprints, fn, norm)


def bin_accuracy(yp, yt):
    return torch.mean((torch.round(yp) == yt).to(torch.float))


def average_support(fingerprints, fn):
    t_fingerprints = torch.Tensor(fingerprints).cuda()

    def fn_br(yp,yt):
        yp_b, yt_b = torch.broadcast_tensors(yp, yt)
        return fn(yp_b, yt_b.detach())

    def g(yp, yt):
        # correct distance
        p_dist = fn_br(yp, yt)

        c = torch.empty(yp.shape[0])
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = fn_br(yp[i].unsqueeze(0), t_fingerprints)

            # shift distance so bad examples are positive
            dist -= p_dist[i]
            dist *= -1

            dist_n = F.sigmoid(dist)

            c[i] = torch.mean(dist_n)

        score = torch.mean(c)

        return score

    return g


def average_support_mse(fingerprints):
    def fn(yp, yt):
        return torch.sum((yp - yt) ** 2, axis=1)
    return average_support(fingerprints, fn)


def average_support_weighted(fingerprints, fn):
    t_fingerprints = torch.Tensor(fingerprints).cuda()

    def fn_br(yp,yt,att):
        yp_b = yp.expand(yt.shape)
        att_b = att.expand(yt.shape)

        return fn(yp_b, yt.detach(), att_b)

    def g(yp, yt, att):
        # correct distance
        p_dist = fn_br(yp, yt, att)

        c = torch.empty(yp.shape[0])
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = fn_br(yp[i].unsqueeze(0), t_fingerprints, att[i].unsqueeze(0))

            # shift distance so bad examples are positive
            dist -= p_dist[i]
            dist *= -1

            dist_n = F.sigmoid(dist)

            c[i] = torch.mean(dist_n)

        score = torch.mean(c)

        return score

    return g


def average_support_weighted_mse(fingerprints):
    def fn(yp, yt, att):

        l = (yp - yt) ** 2
        l *= att
        return torch.sum(l, axis=1)
    return average_support_weighted(fingerprints, fn)


def average_position_weighted(fingerprints, fn, norm=True):
    t_fingerprints = torch.Tensor(fingerprints).cuda()

    def fn_br(yp,yt,att):
        yp_b = yp.expand(yt.shape)
        att_b = att.expand(yt.shape)

        return fn(yp_b.detach(), yt.detach(), att_b.detach())

    def g(yp, yt, att):
        # correct distance
        p_dist = fn_br(yp, yt, att)

        c = torch.empty(yp.shape[0])
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = fn_br(yp[i].unsqueeze(0), t_fingerprints, att[i].unsqueeze(0))

            # number of fragment that are closer or equal
            count = torch.sum((dist <= p_dist[i]).to(torch.float))

            if norm:
                count /= t_fingerprints.shape[0]

            c[i] = count
            
        score = torch.mean(c)

        return score

    return g


def average_position_weighted_mse(fingerprints, norm):
    def fn(yp, yt, att):
        l = (yp - yt) ** 2
        l *= att
        return torch.sum(l, axis=1)
    return average_position_weighted(fingerprints, fn, norm)


def top_k_acc(t_fingerprints, fn, k, pre=''):

    def fn_br(yp,yt):
        yp_b, yt_b = torch.broadcast_tensors(yp, yt)
        return fn(yp_b.detach(), yt_b.detach())

    def g(yp, yt):
        # correct distance
        p_dist = fn_br(yp, yt)

        c = torch.empty(yp.shape[0], len(k))
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = fn_br(yp[i].unsqueeze(0), t_fingerprints)

            # number of fragment that are closer or equal
            count = torch.sum((dist < p_dist[i]).to(torch.float))

            for j in range(len(k)):
                c[i,j] = int(count < k[j])
            
        score = torch.mean(c, axis=0)

        m = {'%sacc_%d' % (pre,h):v.item() for h,v in zip(k,score)}

        return m

    return g


def top_k_acc_mse(t_fingerprints, k, pre):
    def fn(yp, yt):
        return torch.sum((yp - yt) ** 2, axis=1)
    return top_k_acc(t_fingerprints, fn, k, pre)


class Model(object):
    def __init__(self):
        pass

    def setup_parser(self, name, parser):
        pass

    def build_model(self, args):
        pass

    def get_metrics(self, args, std, mean, train_dat, test_dat):
        '''Returns a dict of metrics'''
        pass

    def get_train_mode(self):
        pass


class VoxelNet(Model):

    def setup_parser(self, name, parser):
        sub = parser.add_parser(name)
        sub.add_argument('--in_channels', type=int, default=18)
        sub.add_argument('--output_size', type=int, default=2048)
        sub.add_argument('--pad', default=False, action='store_true')
        sub.add_argument('--blocks', nargs='+', type=int, default=[32,64])
        sub.add_argument('--fc', nargs='+', type=int, default=[2048])

        sub.add_argument('--use_all_labels', default=False, action='store_true')

    def build_model(self, args):
        m = VoxelFingerprintNet2b(
            in_channels=args.in_channels,
            output_size=args.output_size,
            blocks=args.blocks,
            fc=args.fc,
            pad=args.pad
        ).cuda()
        return m

    def get_metrics(self, args, std, mean, train_dat, test_dat):
        
        all_fp = list(set(train_dat.valid_fingerprints + test_dat.valid_fingerprints))
        fingerprints = train_dat.fingerprints['fingerprint_data'][all_fp]

        train_fingerprints = train_dat.fingerprints['fingerprint_data'][train_dat.valid_fingerprints]
        test_fingerprints = train_dat.fingerprints['fingerprint_data'][test_dat.valid_fingerprints]

        t_all_fp = torch.Tensor(fingerprints).cuda()
        t_test_fp = torch.Tensor(test_fingerprints).cuda()

        metrics = {
            'acc': top_k_acc_mse(t_all_fp, k=[1,5,10,50,100], pre='all'),
            'acc2': top_k_acc_mse(t_test_fp, k=[1,5,10,50,100], pre='test'),
        }

        POS_METRICS = [
            ('mse', average_position_mse),
        ]

        for name, eval_fn in POS_METRICS:
            metrics.update({
                'pos_%s' % name: eval_fn(fingerprints, norm=True),
                'pos_%s_raw' % name: eval_fn(fingerprints, norm=False),
            })

        return metrics

    def get_loss(self, args, std, mean, train_dat, test_dat):
        
        all_fp = set(train_dat.valid_fingerprints)
        if args.use_all_labels:
            all_fp |= set(test_dat.valid_fingerprints)
        all_fp = list(all_fp)

        fingerprints = train_dat.fingerprints['fingerprint_data'][all_fp]

        loss_fn = average_support_mse(fingerprints)

        return loss_fn

    def get_train_mode(self):
        return train

class Voxel3(Model):

    def setup_parser(self, name, parser):
        sub = parser.add_parser(name)
        sub.add_argument('--in_channels', type=int, default=18)
        sub.add_argument('--in_fingerprint', type=int, default=2048)
        sub.add_argument('--f1', type=int, default=32)
        sub.add_argument('--f2', type=int, default=64)
        sub.add_argument('--r1', type=int, default=256)
        sub.add_argument('--r2', type=int, default=256)
        sub.add_argument('--p1', type=int, default=256)

    def build_model(self, args):
        m = VoxelFingerprintNet3(
            in_channels=args.in_channels,
            in_fingerprint=args.in_fingerprint,
            f1=args.f1,
            f2=args.f2,
            r1=args.r1,
            r2=args.r2,
            p1=args.p1,
        ).cuda()
        return m

    def get_metrics(self, args, std, mean, train_dat, test_dat):
        metrics = {
            'bin_acc': bin_accuracy,
        }
        return metrics

    def get_train_mode(self):
        return train_dual

class Voxel4(Model):

    def setup_parser(self, name, parser):
        sub = parser.add_parser(name)
        sub.add_argument('--in_channels', type=int, default=18)
        sub.add_argument('--output_size', type=int, default=2048)
        sub.add_argument('--sigmoid', default=False, action='store_true')
        sub.add_argument('--batchnorm', default=False, action='store_true')
        sub.add_argument('--f1', type=int, default=32)
        sub.add_argument('--f2', type=int, default=64)

    def build_model(self, args):
        m = VoxelFingerprintNet4(
            in_channels=args.in_channels,
            output_size=args.output_size,
            sigmoid=args.sigmoid,
            batchnorm=args.batchnorm,
            f1=args.f1,
            f2=args.f2
        ).cuda()
        return m

    def get_metrics(self, args, std, mean, train_dat, test_dat):
        
        all_fp = list(set(train_dat.valid_fingerprints + test_dat.valid_fingerprints))
        fingerprints = train_dat.fingerprints['fingerprint_data'][all_fp]

        train_fingerprints = train_dat.fingerprints['fingerprint_data'][train_dat.valid_fingerprints]
        test_fingerprints = train_dat.fingerprints['fingerprint_data'][test_dat.valid_fingerprints]

        metrics = {}

        POS_METRICS = [
            ('mse_att', average_position_weighted_mse),
        ]

        for name, eval_fn in POS_METRICS:
            metrics.update({
                'pos_%s' % name: eval_fn(fingerprints, norm=True),
                'pos_%s_raw' % name: eval_fn(fingerprints, norm=False),
            })

        def avg_att(yp,yt,att):
            return att.cpu().detach().numpy()

        metrics.update({
            'attention': avg_att
        })

        return metrics

    def get_loss(self, args, std, mean, train_dat, test_dat):
        
        all_fp = list(set(train_dat.valid_fingerprints + test_dat.valid_fingerprints))
        fingerprints = train_dat.fingerprints['fingerprint_data'][all_fp]

        loss_fn = average_support_weighted_mse(fingerprints)

        return loss_fn

    def get_train_mode(self):
        return train_attention

class Latent1(Model):

    def setup_parser(self, name, parser):
        sub = parser.add_parser(name)
        sub.add_argument('frag_loss')
        sub.add_argument('context_loss')

    def build_model(self, args):
        encoder = LatentEncoder(9, True).cuda()
        decoder = LatentDecoder(9, True).cuda()
        context_encoder = LatentEncoder(18, True).cuda()
        return (encoder, decoder, context_encoder)

    def get_metrics(self, args, std, mean, train_dat, test_dat):
        def average_reconstruction(reconstructed, t_frag, z, z2):
            return torch.mean(reconstructed)

        def average_frag(reconstructed, t_frag, z, z2):
            return torch.mean(t_frag)

        return {
            'average_reconstruction': average_reconstruction,
            'average_frag': average_frag
        }

    def get_train_mode(self):
        return train_latent

class Skip1(Model):

    def setup_parser(self, name, parser):
        sub = parser.add_parser(name)
        sub.add_argument('--input_channels',type=int,default=18)
        sub.add_argument('--output_channels',type=int,default=9)
        sub.add_argument('--frag_loss',default='mse')

    def build_model(self, args):
        m = FullSkipV1(
            args.input_channels, 
            args.output_channels
        ).cuda()
        return m

    def get_metrics(self, args, std, mean, train_dat, test_dat):
        def average_gt(yp,yt):
            return torch.mean(yt)

        def average_pred(yp,yt):
            return torch.mean(yp)

        return {
            'average_gt': average_gt,
            'average_pred': average_pred
        }

    def get_train_mode(self):
        return train_latent_direct


MODELS = {
    'voxelnet': VoxelNet(),
    'voxel3': Voxel3(),
    'voxel4': Voxel4(),
    'latent1': Latent1(),
    'skip1': Skip1(),
}
