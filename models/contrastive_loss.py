import logging
from itertools import combinations, permutations, product
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel, BertConfig, ViTConfig
import torchvision.models as models


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.3, margin2=0.99, temp=1.0):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.margin2 = margin2
        self.temp = temp

    def forward(self, labels, img_emb, text_emb):
        labels = labels.cpu().data.numpy()
        # all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = np.array(list(permutations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        weak_positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        # neg_nums = len(positive_pairs) * 16 if len(negative_pairs) > len(positive_pairs) * 16 else len(negative_pairs)
        # neg_nums = len(img_emb) if len(negative_pairs) > len(img_emb) else len(negative_pairs)
        # negative_pairs = negative_pairs[0:neg_nums]

        if img_emb.is_cuda:
            weak_positive_pairs = weak_positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        positive_loss = 1 - F.cosine_similarity(img_emb, text_emb)
        # weak_positive_sims = F.cosine_similarity(img_emb[weak_positive_pairs[:, 0]],
        #                                          text_emb[weak_positive_pairs[:, 1]])
        # weak_positive_loss = F.relu(weak_positive_sims - self.margin2) + 0.1 * (1 - weak_positive_sims)

        negative_loss = F.relu(
            F.cosine_similarity(img_emb[negative_pairs[:, 0]], text_emb[negative_pairs[:, 1]]) - self.margin)
        con_loss = torch.cat([positive_loss, negative_loss], dim=0).mean()

        return con_loss


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.KL = nn.KLDivLoss(reduction='none')

    def forward(self, preds, img_logits, text_logits):
        preds = F.softmax(preds, dim=1)
        img_preds = F.softmax(img_logits, dim=1)
        text_preds = F.softmax(text_logits, dim=1)
        # KD_loss = 0.5 * self.KL(input=torch.log(text_preds), target=preds).mean() + 0.5 * self.KL(
        #     input=torch.log(img_preds), target=preds).mean()
        # KD_loss = 0.5 * self.KL(input=torch.log(text_preds), target=img_preds).mean() + 0.5 * self.KL(
        #     input=torch.log(img_preds), target=text_preds).mean()
        KD_loss = (0.9 * self.KL(input=torch.log(text_preds), target=preds).sum(dim=1) + 0.1 * self.KL(
            input=torch.log(text_preds), target=img_preds).sum(dim=1)).mean() + \
                  (0.9 * self.KL(input=torch.log(img_preds), target=preds).sum(dim=1) + 0.1 * self.KL(
                      input=torch.log(img_preds), target=text_preds).sum(dim=1)).mean()

        loss = KD_loss

        return loss


def JSDiv(p, q):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss()
    p = F.softmax(p.view(-1, p.shape[-1]), dim=-1)
    q = F.softmax(q.view(-1, q.shape[-1]), dim=-1)
    log_mean_output = ((p + q) / 2).log()
    return (KLDivLoss(log_mean_output, p) + KLDivLoss(log_mean_output, q)) / 2


class JSLoss(nn.Module):
    def __init__(self):
        super(JSLoss, self).__init__()
        self.KLDivLoss = nn.KLDivLoss()

    def forward(self, preds, img_logits, text_logits):
        preds = F.softmax(preds, dim=1)
        img_preds = F.softmax(img_logits, dim=1)
        text_preds = F.softmax(text_logits, dim=1)
        log_mean_output = ((img_preds + text_preds) / 2).log()
        jsd_loss = self.KLDivLoss(log_mean_output, img_preds) + self.KLDivLoss(log_mean_output, text_preds)
        kl_loss = self.KLDivLoss(torch.log(img_preds), preds) + self.KLDivLoss(torch.log(text_preds), preds)

        return (jsd_loss + kl_loss).mean()


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, preds, img_logits, text_logits):
        loss = 0.5 * (1 - F.cosine_similarity(img_logits, preds)).mean() + 0.5 * (
                1 - F.cosine_similarity(text_logits, preds)).mean()
        return loss


class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()
        self.MMD = MMD()

    def forward(self, preds, img_logits, text_logits):
        preds = F.softmax(preds, dim=1)
        img_preds = F.softmax(img_logits, dim=1)
        text_preds = F.softmax(text_logits, dim=1)
        MMD_loss = 0.5 * self.MMD(source=img_preds, target=preds).mean() + 0.5 * self.MMD(source=text_preds,
                                                                                          target=preds).mean()
        return MMD_loss


class MMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self, img_logits, text_logits):
        mse_loss = self.MSE(text_logits, img_logits)

        loss = mse_loss.mean()
        return loss


class InterContrastiveLoss(nn.Module):
    def __init__(self, margin=0.7):
        super(InterContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, labels, img_embs, text_embs):
        bs = img_embs.shape[0]
        pos_dist = 1 - F.cosine_similarity(img_embs, text_embs)

        img_embs = F.normalize(img_embs, p=2, dim=1)
        text_embs = F.normalize(text_embs, p=2, dim=1)

        all_dist = 1 - torch.mm(img_embs, text_embs.t())
        neg_mask = ((labels.expand(bs, bs).eq(labels.expand(bs, bs).t())) != 1).long()
        # wp_mask = 1 - neg_mask - torch.eye(bs, device=img_embs.device)

        pos_loss = pos_dist
        neg_dist = (all_dist * neg_mask).reshape(bs ** 2, -1)
        neg_dist = neg_dist[neg_dist != 0]
        neg_loss = F.relu(self.margin - neg_dist)

        # wp_dist = (all_dist * wp_mask).reshape(bs ** 2, -1)
        # wp_dist = wp_dist[wp_dist != 0]
        # wp_loss = F.relu(0.1 - wp_dist) + F.relu(self.margin - wp_dist)
        # wp_loss = wp_loss[wp_loss != 0]
        # print("num_neg: {}, num_pos: {}".format(len(neg_loss), len(pos_loss)))
        loss = torch.cat([pos_loss, neg_loss], dim=0).mean()

        # neg_dist = all_dist * neg_mask
        # neg_dist = (1 - neg_mask) * 100 + neg_dist  # replace zero with 100
        # neg_dist = torch.min(neg_dist, dim=1)[0]
        # wp_dist = all_dist * wp_mask
        # wp_dist = torch.max(wp_dist, dim=1)[0]
        # wp_loss = F.relu(wp_dist - neg_dist).mean()
        # loss = loss + 0.1 * wp_loss

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.6):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin, reduction='none')

    def forward(self, labels, img_embs, text_embs):
        bs = img_embs.shape[0]
        pos_dist = 1 - F.cosine_similarity(img_embs, text_embs)

        img_embs = F.normalize(img_embs, p=2, dim=1)
        text_embs = F.normalize(text_embs, p=2, dim=1)

        all_dist = 1 - torch.mm(img_embs, text_embs.t())
        neg_mask = ((labels.expand(bs, bs).eq(labels.expand(bs, bs).t())) != 1).long()
        # wp_mask = 1 - neg_mask - torch.eye(bs, device=img_embs.device)

        y = torch.ones_like(pos_dist)
        neg_dist = all_dist * neg_mask
        neg_dist = (1 - neg_mask) * 100 + neg_dist  # replace zero with 100

        # only hardest negative samples
        # it_neg_dist = torch.min(neg_dist, dim=1)[0]
        # ti_neg_dist = torch.min(neg_dist, dim=0)[0]
        # all negative samples
        # it_neg_dist = torch.sum(neg_dist, dim=1) / torch.count_nonzero(neg_dist, dim=1)
        # ti_neg_dist = torch.sum(neg_dist, dim=0) / torch.count_nonzero(neg_dist, dim=0)

        # it_loss = self.ranking_loss(it_neg_dist, pos_dist, y)
        # ti_loss = self.ranking_loss(ti_neg_dist, pos_dist, y)
        # loss = torch.cat([it_loss, ti_loss], dim=0).mean()

        neg_dist = torch.min(neg_dist, dim=1)[0]
        loss = self.ranking_loss(neg_dist, pos_dist, y).mean()

        return loss


class SelectiveTripletsLoss(nn.Module):
    def __init__(self, margin1=0.6, margin2=0.1):
        super(SelectiveTripletsLoss, self).__init__()
        self.ranking_loss1 = nn.MarginRankingLoss(margin=margin1, reduction='none')
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin2, reduction='none')
        # self.margins = nn.Parameter(torch.ones([num_labels, num_labels])*0.8)

    def forward(self, labels, img_embs, text_embs):
        bs = img_embs.shape[0]
        pos_dist = 1 - F.cosine_similarity(img_embs, text_embs)

        img_embs = F.normalize(img_embs, p=2, dim=1)
        text_embs = F.normalize(text_embs, p=2, dim=1)

        all_dist = 1 - torch.mm(img_embs, text_embs.t())
        neg_mask = ((labels.expand(bs, bs).eq(labels.expand(bs, bs).t())) != 1).long()
        wp_mask = 1 - neg_mask - torch.eye(bs, device=img_embs.device)

        # let Min(neg_dist) - pos_dist > alpha
        # use only hardest negative samples
        y = torch.ones_like(pos_dist)
        neg_dist = all_dist * neg_mask
        neg_dist = (1 - neg_mask) * 100 + neg_dist  # replace zero with 100
        neg_dist, neg_inx = torch.min(neg_dist, dim=1)
        neg_labels = labels[neg_inx]
        # weights = self.margins[labels, neg_labels]
        neg_loss = self.ranking_loss1(neg_dist, pos_dist, y).mean()

        # let Max(wp_dist) - pos_dist < beta
        wp_dist = all_dist * wp_mask
        nz = torch.count_nonzero(wp_dist, dim=1)
        if nz.sum() == 0:
            loss = neg_loss
            return loss
        nz_pos_dist = pos_dist[nz != 0]
        nz_y = torch.ones_like(nz_pos_dist)
        nz_wp_dist, nz_wp_inx = torch.max(wp_dist[nz != 0], dim=1)
        wp_loss = self.ranking_loss2(nz_wp_dist, nz_pos_dist, nz_y).mean()
        # wp_loss = self.ranking_loss2(neg_dist[nz != 0], nz_wp_dist, nz_y).mean()
        # loss = neg_loss + wp_loss
        loss = neg_loss
        # loss = torch.cat([neg_loss, wp_loss], dim=0).mean()

        # show the distance between Max(wp_dist) and Min(neg_dist)
        # print(neg_dist[nz != 0]-nz_wp_dist)

        return loss


class AllTripletsLoss(nn.Module):
    def __init__(self, margin1=0.6, margin2=0.1):
        super(AllTripletsLoss, self).__init__()
        self.ranking_loss1 = nn.MarginRankingLoss(margin=margin1, reduction='none')
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin2, reduction='none')

    def forward(self, labels, img_embs, text_embs):
        bs = img_embs.shape[0]
        pos_dist = 1 - F.cosine_similarity(img_embs, text_embs)

        img_embs = F.normalize(img_embs, p=2, dim=1)
        text_embs = F.normalize(text_embs, p=2, dim=1)

        all_dist = 1 - torch.mm(img_embs, text_embs.t())
        neg_mask = ((labels.expand(bs, bs).eq(labels.expand(bs, bs).t())) != 1).long()
        wp_mask = 1 - neg_mask - torch.eye(bs, device=img_embs.device)

        # let Min(neg_dist) - pos_dist > alpha
        # use only hardest negative samples
        y = torch.ones_like(pos_dist)
        neg_dist = all_dist * neg_mask

        neg_dist = torch.mean(neg_dist, dim=1)
        neg_loss = self.ranking_loss1(neg_dist, pos_dist, y).mean()

        # let Max(wp_dist) - pos_dist < beta
        wp_dist = all_dist * wp_mask
        nz = torch.count_nonzero(wp_dist, dim=1)
        if nz.sum() == 0:
            loss = neg_loss
            return loss
        nz_pos_dist = pos_dist[nz != 0]
        nz_y = torch.ones_like(nz_pos_dist)
        # nz_wp_dist = torch.max(wp_dist[nz != 0], dim=1)[0]
        nz_wp_dist = torch.mean(wp_dist[nz != 0], dim=1)
        wp_loss = self.ranking_loss2(nz_wp_dist, nz_pos_dist, nz_y).mean()
        # wp_loss = F.relu((nz_wp_dist - nz_pos_dist) - self.margin2).mean()
        loss = neg_loss + 0.1 * wp_loss
        # loss = neg_loss
        # loss = torch.cat([neg_loss, wp_loss], dim=0).mean()
        return loss
