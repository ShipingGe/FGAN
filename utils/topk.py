import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

import torch
import torch.nn as nn


class PerturbedTopK(nn.Module):
    def __init__(self, num_samples: int = 1000, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma

    def __call__(self, x, k):
        return PerturbedTopKFunction.apply(x, k, self.num_samples, self.sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        # b, nS, k, d
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 4)

        noise_gradient = ctx.noise
        expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 3)


def built_topk_selectors(input_len, pooled_len):
    # Build our selector
    our_topk = TopKOperator()
    cfg = TopKConfig(input_len=input_len,
                     pooled_len=pooled_len,
                     flip_right=True,
                     sort_back=False,
                     iterative=False,
                     base=20,
                     )
    our_topk.set_config(cfg)
    # Build baseline (iterative) selector
    iter_topk = TopKOperator()
    cfg = TopKConfig(input_len=input_len,
                     pooled_len=pooled_len,
                     flip_right=True,
                     sort_back=False,
                     iterative=True,
                     base=-1,
                     )
    iter_topk.set_config(cfg)
    return our_topk, iter_topk


class TopKConfig:
    input_len: int = -1
    pooled_len: int = -1
    depth: int = 0
    flip_right: bool = True
    sort_back: bool = False
    iterative: int = 0
    base: int = 20

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TopKOperator(nn.Module):
    """Funny no-net implementation xD"""

    def __init__(self):
        super(TopKOperator, self).__init__()
        self.iterations_performed = 0

    def set_config(self, pooler_config):
        self.input_len = pooler_config.input_len
        self.pooled_len = pooler_config.pooled_len
        self.depth = pooler_config.depth if pooler_config.depth > 0 \
            else int(
            torch.log2(torch.tensor(self.input_len / self.pooled_len)))
        self.flip_right = pooler_config.flip_right
        self.sort_back = pooler_config.sort_back

        self.iterative = pooler_config.iterative
        self.base = pooler_config.base
        self.name = 'iter_topk' if self.iterative else 'our_topk'

    def forward(self, embs, scores):
        """
        embs: batch x input_len x emb_depth
        scores: batch x input_len x 1
        """
        new_embs = []
        new_scores = []
        # pad embeddings with zeros
        embs, scores = self.pad_to_input_len(self.input_len, embs, scores)

        if self.iterative:
            new_embs, new_scores = self.vectorized_iterative_topk(embs, scores)
            return new_embs, new_scores

        for batch_i in range(embs.shape[0]):
            embs_tmp, scores_tmp = self.our_topk(embs[batch_i].unsqueeze(0), scores[batch_i].unsqueeze(0))
            assert len(embs_tmp.shape) == 3 and embs_tmp.shape[0] == 1
            assert len(scores_tmp.shape) == 2 and scores_tmp.shape[0] == 1

            new_embs.append(embs_tmp)
            new_scores.append(scores_tmp)
        new_embs = torch.cat(new_embs, dim=0)
        new_scores = torch.cat(new_scores, dim=0)

        return new_embs, new_scores

    @staticmethod
    def pad_to_input_len(input_len, embs, scores):
        sh = list(embs.shape)
        sh[1] = input_len - sh[1]
        assert sh[1] >= 0
        emb_pad = torch.zeros(sh, dtype=embs.dtype, device=embs.device)
        embs = torch.cat((embs, emb_pad), dim=1)
        # pad scores with negative big score
        sh = list(scores.shape)
        sh[1] = input_len - sh[1]
        score_pad = torch.zeros(sh, dtype=scores.dtype, device=scores.device) + 0.00001
        scores = torch.cat((scores, score_pad), dim=1).squeeze(2)
        return embs, scores

    def our_topk(self, embs, scores):
        """This is an implementation of our topk function"""
        e = embs.shape[2]
        s = partial(F.softmax, dim=1)
        target_size = self.input_len // 2
        for layer in range(self.depth):
            pairs_idx = self.get_topk_pair_idx(scores)  # firstly, sort by scores and 'draw' pairs
            scores_before = scores.clone()
            scores_converged = scores[:, pairs_idx]
            if self.base > 0:
                exped = torch.pow(self.base, scores_converged)  # exponentiation with any given base
                scores_converged = s(exped)  # softmax over scores (the more it converges usually the better)
            else:
                raise ValueError
            scores = (scores_before[:, pairs_idx] * scores_converged) \
                .sum(dim=1)  # new scores are a linear interpolation in pairs provided
            embs = (embs[:, pairs_idx] * scores_converged.unsqueeze(3)
                    .expand((1, 2, target_size, e))) \
                .sum(dim=1)  # new embedding are also linearly interpolated from the old pair elements

            # De-sort back into chunk-positions
            # (this may be useful if we want to have an old ordering
            # of top-k elements in the sequence)
            if self.sort_back:
                scores = scores[:, pairs_idx[0].sort().indices]
                embs = embs[:, pairs_idx[0].sort().indices]

            # Finish the round with new target assignments
            current_size = target_size
            target_size = embs.shape[1] // 2

            if current_size < self.pooled_len:
                break
        return embs, scores

    def get_topk_pair_idx(self, scores):
        """ Sort by value and fold.
        This is halving the number of inputs in each step.
        This keeps topk token in different sampling 'pool'
        """
        sort_idx = scores.sort(descending=True).indices

        l_half = sort_idx.shape[-1] // 2
        left = sort_idx[:, :l_half]
        right = sort_idx[:, l_half:]
        if self.flip_right:
            right = torch.flip(right, dims=(1, 0))
        pairs_idx = torch.cat((left, right),
                              dim=0)
        return pairs_idx

    def vectorized_iterative_topk(self, embs, scores):
        """Iterative approach to test as a baseline"""
        new_scores = []
        new_embs = []
        max_weights = []  # debug, and proving that this is not sharply defined
        alpha = 1.0
        bs, tlen, hdim = embs.shape
        for i in range(self.pooled_len):
            miv = scores.max(dim=1)
            m = miv.values
            squared_dist = -(scores - m.unsqueeze(1)) ** 2
            weights = F.softmax(squared_dist * alpha, dim=1)
            ith_vec = (weights.unsqueeze(2) * embs).sum(1)
            weighted_scores = weights * scores
            ith_score = weighted_scores.sum(1)
            max_ith_weight = weights.max(1)
            new_embs.append(ith_vec)
            new_scores.append(ith_score)
            max_weights.append(max_ith_weight.values)
            for i, el in enumerate(miv.indices):
                scores[i, el] = -10000

        stacked_max_ith_weights = torch.stack(
            max_weights)  # look here to check how poorly designed is this approximation
        stacked_embs = torch.stack(new_embs).permute(1, 0, 2)
        stacked_scores = torch.stack(new_scores).permute(1, 0)

        self.iterations_performed += 1
        if self.iterations_performed % 1000 == 0:
            print(f'Iterative topk: \n \t Cosine similarity is : '
                  f'{torch.cosine_similarity(stacked_embs[0, 0], stacked_embs[0, -1], dim=0)}')
            print(f'\t Maximal weight of a single vector is : {stacked_max_ith_weights.max()}\n')
        assert stacked_embs.shape[2] == embs.shape[2]
        assert stacked_embs.shape[1] == self.pooled_len
        return stacked_embs, stacked_scores


def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    """standard forward of sinkhorn."""

    bs, _, k_ = C.size()

    v = torch.ones([bs, 1, k_], device=C.device) / (k_)
    G = torch.exp(-C / epsilon)

    for _ in range(max_iter):
        u = mu / (G * v).sum(-1, keepdim=True)
        v = nu / (G * u).sum(-2, keepdim=True)

    Gamma = u * G * v
    return Gamma


def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    """sinkhorn forward in log space."""

    bs, n, k_ = C.size()
    k = k_ - 1

    f = torch.zeros([bs, n, 1], device=C.device)
    g = torch.zeros([bs, 1, k + 1], device=C.device)

    epsilon_log_mu = epsilon * torch.log(mu)
    epsilon_log_nu = epsilon * torch.log(nu)

    def min_epsilon_row(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

    def min_epsilon_col(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

    for _ in range(max_iter):
        f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

    Gamma = torch.exp((-C + f + g) / epsilon)
    return Gamma


def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    nu_ = nu[:, :, :-1]
    Gamma_ = Gamma[:, :, :-1]

    bs, n, k_ = Gamma.size()

    inv_mu = 1. / (mu.view([1, -1]))  # [1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) \
            - torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)  # [bs, k, k]

    inv_Kappa = torch.inverse(Kappa)  # [bs, k, k]

    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = Gamma_mu.matmul(inv_Kappa)  # [bs, n, k]
    G1 = grad_output_Gamma * Gamma  # [bs, n, k+1]

    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma  # [bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L)  # [bs, 1, k]
    G22 = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma  # [bs, n, k+1]
    G23 = - F.pad(g1_L, pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G2 = G21 + G22 + G23  # [bs, n, k+1]

    del g1, G21, G22, G23, Gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1)  # [bs, k+1, 1]
    g2 = g2[:, :-1, :]  # [bs, k, 1]
    G31 = - L.matmul(g2) * Gamma  # [bs, n, k+1]
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G3 = G31 + G32  # [bs, n, k+1]
    #            del g2, G31, G32, L

    grad_C = (-G1 + G2 + G3) / epsilon  # [bs, n, k+1]

    return grad_C


import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
class TopKFunc1(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    print('Nan appeared in Gamma, re-computing...')
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon

        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=100):
        super(TopK_custom, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0, 1]).view([1, 1, 2])
        self.max_iter = max_iter

    def forward(self, scores):
        anchors = self.anchors.to(scores.device)
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - anchors) ** 2
        C = C / (C.max().detach())
        # print(C)
        mu = torch.ones([1, n, 1], requires_grad=False, device=scores.device) / n
        nu = torch.FloatTensor([self.k / n, (n - self.k) / n]).view([1, 1, 2]).to(scores.device)

        Gamma = TopKFunc1.apply(C, mu, nu, self.epsilon, self.max_iter)
        # print(Gamma)
        A = Gamma[:, :, 0] * n

        return A