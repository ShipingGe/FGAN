import logging
from itertools import combinations, permutations, product
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel, BertConfig, ViTConfig
import torchvision.models as models


class FGCL(nn.Module):
    def __init__(self, margin1=0.5, margin2=0.1, wp_weight=0.1):
        super(FGCL, self).__init__()
        # self.ranking_loss1 = nn.MarginRankingLoss(margin=margin1, reduction='none')
        # self.ranking_loss2 = nn.MarginRankingLoss(margin=margin2, reduction='none')
        self.pdist = nn.PairwiseDistance(p=2)
        self.margin1 = margin1
        self.margin2 = margin2
        self.wp_weight = wp_weight

    def forward(self, labels, img_embs, text_embs):
        bs = img_embs.shape[0]
        # pos_dist = 1 - F.cosine_similarity(img_embs, text_embs)

        img_embs = F.normalize(img_embs, p=2, dim=-1)
        text_embs = F.normalize(text_embs, p=2, dim=-1)

        all_dist = 1 - torch.mm(img_embs, text_embs.t())
        pos_dist = torch.diag(all_dist)
        # ids = list(product(np.arange(bs), repeat=2))
        # ids = torch.tensor(ids, device=img_embs.device)
        # all_dist = self.pdist(img_embs[ids[:, 0]], text_embs[ids[:, 1]]).reshape(bs, -1)

        neg_mask = ((labels.expand(bs, bs).eq(labels.expand(bs, bs).t())) != 1).long()
        wp_mask = 1 - neg_mask - torch.eye(bs, device=img_embs.device)

        # let Min(neg_dist) - pos_dist > alpha
        # use only hardest negative samples
        y = torch.ones_like(pos_dist)
        neg_dist = all_dist * neg_mask + (1 - neg_mask) * 100
        neg_dist, neg_inx = torch.min(neg_dist, dim=1)
        # neg_loss = self.ranking_loss1(neg_dist, pos_dist, y).mean()
        neg_loss = F.relu(pos_dist - neg_dist + self.margin1).mean()

        # let Max(wp_dist) - pos_dist < beta
        wp_dist = all_dist * wp_mask
        nz = torch.count_nonzero(wp_dist, dim=1)
        if nz.sum() == 0:
            loss = neg_loss
            return loss
        nz_pos_dist = pos_dist[nz != 0]
        nz_neg_dist = neg_dist[nz != 0]
        nz_y = torch.ones_like(nz_pos_dist)
        nz_wp_dist_max, _ = torch.max(wp_dist[nz != 0], dim=1)
        # wp_loss = self.ranking_loss2(nz_wp_dist, nz_pos_dist, nz_y).mean()
        wp_loss = F.relu(nz_pos_dist - nz_wp_dist_max + self.margin2).mean()
        # wp_loss = F.relu(nz_wp_dist_max - nz_neg_dist + self.margin2).mean()

        # nz_wp_dist_min, _ = torch.min(wp_dist[nz != 0], dim=1)
        # wp_loss = F.relu(nz_pos_dist - nz_wp_dist_min + self.margin2).mean()

        loss = neg_loss + self.wp_weight * wp_loss
        # loss = neg_loss

        return loss
