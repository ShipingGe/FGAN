import torch
import torch.nn as nn

import torch.nn.functional as F

class FGCL(nn.Module):
    def __init__(self, margin1=0.5, margin2=0.1, wp_weight=0.1):
        super(FGCL, self).__init__()

        self.pdist = nn.PairwiseDistance(p=2)
        self.margin1 = margin1
        self.margin2 = margin2
        self.wp_weight = wp_weight

    def forward(self, labels, img_embs, text_embs):
        bs = img_embs.shape[0]
        img_embs = F.normalize(img_embs, p=2, dim=-1)
        text_embs = F.normalize(text_embs, p=2, dim=-1)

        all_dist = 1 - torch.mm(img_embs, text_embs.t())
        pos_dist = torch.diag(all_dist)

        neg_mask = ((labels.expand(bs, bs).eq(labels.expand(bs, bs).t())) != 1).long()
        wp_mask = 1 - neg_mask - torch.eye(bs, device=img_embs.device)

        neg_dist = all_dist * neg_mask + (1 - neg_mask) * 100
        neg_dist, neg_inx = torch.min(neg_dist, dim=1)

        neg_loss = F.relu(pos_dist - neg_dist + self.margin1).mean()

        # let Max(wp_dist) - pos_dist < beta
        wp_dist = all_dist * wp_mask
        nz = torch.count_nonzero(wp_dist, dim=1)
        if nz.sum() == 0:
            loss = neg_loss
            return loss
        nz_pos_dist = pos_dist[nz != 0]

        nz_wp_dist_max, _ = torch.max(wp_dist[nz != 0], dim=1)

        wp_loss = F.relu(nz_pos_dist - nz_wp_dist_max + self.margin2).mean()

        loss = neg_loss + self.wp_weight * wp_loss

        return loss
