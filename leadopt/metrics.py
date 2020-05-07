
import torch


def tanimoto(yp, yt):
    intersect = torch.sum(yt * torch.round(yp), axis=1)
    union = torch.sum(torch.clamp(yt + torch.round(yp), 0, 1), axis=1)
    return torch.mean(intersect / union)
