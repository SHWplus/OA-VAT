import torch

def pairwise_distance(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist


def NFC(feat: torch.tensor, k1=2, k2=2):
    feat = feat.clone()
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
    dist = pairwise_distance(feat.to('cuda'), feat.to('cuda')).to('cpu')

    eye = torch.eye(dist.size(0)).to(dist.device)
    dist[eye == 1] = 1000
    val, rank = dist.topk(k1, largest=False)
    
    mutual_topk_list = []
    for i in range(rank.size(0)):
        mutual_list = []
        for j in rank[i]:
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)

    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        feat[i] += feat_copy[mutual_topk_list[i]].sum(dim=0)
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
    return feat

