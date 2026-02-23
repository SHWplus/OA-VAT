import torch 
import numpy as np
from torch.nn import functional as F

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

'''
A quantitative metric for Identity density to replce visualization tools 
like t-SNE, which is random and only focus on few samples.
'''
def ID2(feats, pid):
    feats = F.normalize(feats , dim=1, p=2)
    pids = np.asarray(pid)

    id_set = set(pids)
    id_list = list(id_set)
    id_list.sort()
    id_center = []
    for i in id_list:
        mask = pids == i
        x = feats[mask].mean(dim=0)
        id_center.append(x)

    density = torch.zeros(feats.size(0))
    idx = 0
    for i in id_list:
        mask = pids == i
        center = id_center[idx].unsqueeze(0)
        density[mask] = torch.tensor(euclidean_distance(feats[mask], center)).squeeze(1)
        idx += 1
    return density
    