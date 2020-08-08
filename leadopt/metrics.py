
import torch
import torch.nn as nn
import torch.nn.functional as F


def mse(yp, yt):
    """Mean squared error loss."""
    return torch.sum((yp - yt) ** 2, axis=1)


def bce(yp, yt):
    """Binary cross entropy loss."""
    return torch.sum(F.binary_cross_entropy(yp, yt, reduction='none'), axis=1)


def tanimoto(yp, yt):
    """Tanimoto distance metric."""
    intersect = torch.sum(yt * torch.round(yp), axis=1)
    union = torch.sum(torch.clamp(yt + torch.round(yp), 0, 1), axis=1)
    return 1 - (intersect / union)


_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def cos(yp, yt):
    """Cosine distance as a loss (inverted)."""
    return 1 - _cos(yp,yt)


def broadcast_fn(fn, yp, yt):
    """Broadcast a distance function."""
    yp_b, yt_b = torch.broadcast_tensors(yp, yt)
    return fn(yp_b, yt_b)


def average_position(fingerprints, fn, norm=True):
    """Returns the average ranking of the correct fragment relative to all
       possible fragments.
    
    Args:
        fingerprints: NxF tensor of fingerprint data
        fn: distance function to compare fingerprints
        norm: if True, normalize position in range (0,1)
    """
    def _average_position(yp, yt):
        # distance to correct fragment
        p_dist = broadcast_fn(fn, yp, yt.detach())

        c = torch.empty(yp.shape[0])
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = broadcast_fn(fn, yp[i].unsqueeze(0), fingerprints)

            # number of fragment that are closer or equal
            count = torch.sum((dist <= p_dist[i]).to(torch.float))
            c[i] = count
            
        score = torch.mean(c)
        return score

    return _average_position


def average_support(fingerprints, fn):
    """
    """
    def _average_support(yp, yt):
        # correct distance
        p_dist = broadcast_fn(fn, yp, yt)

        c = torch.empty(yp.shape[0])
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = broadcast_fn(fn, yp[i].unsqueeze(0), fingerprints)

            # shift distance so bad examples are positive
            dist -= p_dist[i]
            dist *= -1

            dist_n = torch.sigmoid(dist)

            c[i] = torch.mean(dist_n)

        score = torch.mean(c)

        return score

    return _average_support


def inside_support(fingerprints, fn):
    """
    """
    def _inside_support(yp, yt):
        # correct distance
        p_dist = broadcast_fn(fn, yp, yt)

        c = torch.empty(yp.shape[0])
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = broadcast_fn(fn, yp[i].unsqueeze(0), fingerprints)

            # shift distance so bad examples are positive
            dist -= p_dist[i]
            dist *= -1

            # ignore labels that are further away
            dist[dist < 0] = 0

            dist_n = torch.sigmoid(dist)

            c[i] = torch.mean(dist_n)

        score = torch.mean(c)

        return score

    return _inside_support


def top_k_acc(fingerprints, fn, k, pre=''):
    """Top-k accuracy metric.

    Returns a dict containing top-k accuracies:
    {
        {pre}acc_{k1}: acc_k1,
        {pre}acc_{k2}: acc_k2,
    }

    Args:
        fingerprints: NxF tensor of fingerprints
        fn: distance function to compare fingerprints
        k: List[int] containing K-positions to evaluate (e.g. [1,5,10])
        pre: optional postfix on the metric name
    """

    def _top_k_acc(yp, yt):
        # correct distance
        p_dist = broadcast_fn(fn, yp.detach(), yt.detach())

        c = torch.empty(yp.shape[0], len(k))
        for i in range(yp.shape[0]):
            # compute distance to all other fragments
            dist = broadcast_fn(fn, yp[i].unsqueeze(0).detach(), fingerprints)

            # number of fragment that are closer or equal
            count = torch.sum((dist < p_dist[i]).to(torch.float))

            for j in range(len(k)):
                c[i,j] = int(count < k[j])
            
        score = torch.mean(c, axis=0)
        m = {'%sacc_%d' % (pre, h): v.item() for h,v in zip(k,score)}

        return m

    return _top_k_acc
