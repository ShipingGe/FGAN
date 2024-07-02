# coding=utf-8
import logging
import torch.nn as nn
import torchvision.models as models

from utils.topk import PerturbedTopK



logger = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    def __init__(self, num_t):
        super(TextEncoder, self).__init__()

        self.base = nn.GRU(300, 256, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.linear = nn.Linear(512, 512)
        self.norm = nn.LayerNorm(512)

        self.label_distribution_layer = MeanPoolingLayer(in_channels=512, out_channels=512)

        self.region_embedding_layer = RegionEmbedding(in_channels=512, out_channels=512, k=num_t)

    def forward(self, x, lens):

        x, _ = self.base(x)
        x = self.linear(x)
        x = self.norm(x)

        x_ld, _ = self.label_distribution_layer(x)

        x_re, _ = self.region_embedding_layer(x)

        return x_ld, x_re


class ImageEncoder(nn.Module):
    def __init__(self, num_v):
        super(ImageEncoder, self).__init__()

        self.vgg = models.vgg19_bn(weights='IMAGENET1K_V1')

        for i in self.vgg.parameters():
            i.requires_grad = False

        self.label_distribution_layer = MeanPoolingLayer(in_channels=512, out_channels=512)
        self.region_embedding_layer = RegionEmbedding(in_channels=512, out_channels=512, k=num_v)

    def forward(self, x):

        x = self.vgg.features(x)

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # shape: bs, 14*14, 512
        # x = self.encoder(x)

        x_ld = self.label_distribution_layer(x.permute(0, 2, 1)).squeeze(-1)

        x_re, _ = self.region_embedding_layer(x)

        return x_ld, x_re


class RegionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(RegionEmbedding, self).__init__()
        self.k = k
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, dropout=0.1)
        self.scorer = nn.Sequential(nn.Linear(in_features=in_channels, out_features=1),
                                    nn.Sigmoid())
        self.topk = PerturbedTopK()

        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.ln = nn.LayerNorm(out_channels)


    def forward(self, x):
        attn = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))[0].transpose(0, 1)
        scores = self.scorer(attn).squeeze(-1)

        k = int(self.k * x.shape[1])
        topk_weights = self.topk(scores, k)

        topk_feats = (topk_weights.sum(1).unsqueeze(-1) * x).sum(dim=1) / self.k

        outputs = self.fc(topk_feats)
        outputs = self.ln(outputs)

        return outputs, scores

