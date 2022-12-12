# coding=utf-8
import logging
from itertools import combinations, permutations, product
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel, BertConfig, ViTConfig, \
    ViTForImageClassification
import torchvision.models as models
from .VAE import VanillaVAE
from .GAN import VanillaGAN
from .AutoEncoder import AE, LEAE
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from utils.topk import PerturbedTopK

# from torch_geometric.nn import TopKPooling, MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops
# from torch_geometric.nn import GCNConv, SAGEConv, NNConv
# from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GlobalDiscriminator(nn.Module):
    def __init__(self, global_channels, local_channels):
        super().__init__()
        self.l0 = nn.Linear(global_channels + local_channels, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, global_feats, local_feats):
        h = torch.cat((global_feats, local_feats), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, global_channels, local_channels):
        super().__init__()
        self.c0 = nn.Linear(in_features=global_channels + local_channels, out_features=512)
        self.c1 = nn.Linear(in_features=512, out_features=512)
        self.c2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, global_feats, local_feats):
        global_feats = global_feats.unsqueeze(1).expand(-1, local_feats.shape[1], -1)
        x = torch.cat([local_feats, global_feats], dim=2)
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

        self.base_fc = nn.Sequential(
                                     nn.Linear(in_features=300, out_features=256))

        # self.base_fc = nn.Sequential(nn.LayerNorm(300),
        #                              nn.Linear(in_features=300, out_features=256),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(in_features=256, out_features=256))

        self.global_pooling_layer = AttentionPoolingLayer(in_channels=256, out_channels=512)

        # self.local_pooling_layer = SelectivePoolingLayer(in_channels=256, out_channels=512, k=4)
        self.local_pooling_layer = AttentionPoolingLayer(in_channels=256, out_channels=512)
        # self.local_pooling_layer = MeanPoolingLayer(in_channels=256, out_channels=512)

        self.local_fc = nn.Sequential(nn.LeakyReLU(),
                                      nn.LayerNorm(512),
                                      # nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(in_features=512, out_features=512))

    def forward(self, x):
        x = self.base_fc(x)
        x_global, _ = self.global_pooling_layer(x)

        x_local, scores = self.local_pooling_layer(x)
        x_local = self.local_fc(x_local)

        # with open('text_scores.txt', 'a') as f:
        #     s = scores.cpu().numpy().tolist()
        #     for score in s:
        #         f.write(str(score) + '\n')

        return x_global, x_local


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-3])
        self.vgg = nn.Sequential(*list(models.vgg19_bn(pretrained=True).children())[0][:-1])
        # self.vgg_2 = nn.Sequential(list(models.vgg19_bn(pretrained=True).children())[1])
        # self.vgg_3 = nn.Sequential(list(models.vgg19_bn(pretrained=True).children())[2][:-3])
        # self.sa = nn.MultiheadAttention(embed_dim=512, num_heads=4, dropout=0.1)
        # self.global_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.global_pooling_layer = AttentionPoolingLayer(in_channels=512, out_channels=512)

        # self.local_pooling_layer = MeanPoolingLayer(in_channels=512, out_channels=512)
        # self.local_pooling_layer = SelectivePoolingLayer(in_channels=512, out_channels=512, k=8)
        self.local_pooling_layer = AttentionPoolingLayer(in_channels=512, out_channels=512)
        self.local_fc = nn.Sequential(nn.LeakyReLU(),
                                      nn.LayerNorm(512),
                                      # nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(in_features=512, out_features=512))

    def forward(self, x):
        # x = self.resnet(x)
        # x = self.vgg_2(self.vgg_1(x))
        x = self.vgg(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # shape: bs, 14*14, 512
        # x_global = self.global_pooling_layer(x.permute(0, 2, 1)).squeeze(-1)
        x_global, _ = self.global_pooling_layer(x)

        # x = self.sa(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))[0].transpose(0, 1)
        x_local, scores = self.local_pooling_layer(x)
        x_local = self.local_fc(x_local)
        # with open('img_scores.txt', 'a') as f:
        #     s = scores.cpu().numpy().tolist()
        #     for score in s:
        #         f.write(str(score) + '\n')
        # x_global = self.global_pooling_layer(x)
        # x_global = self.global_fc(self.global_pooling_layer(x.permute(0, 2, 1)).squeeze(-1))
        # x_global = self.global_fc(torch.mean(x, dim=1))

        return x_global, x_local


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.global_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.local_pooling_layer = SelectivePoolingLayer(in_channels=768, out_channels=512, k=2)
        self.local_fc = nn.Sequential(nn.LeakyReLU(),
                                      nn.LayerNorm(512),
                                      nn.Linear(in_features=512, out_features=512))

    def forward(self, text_encoding, text_lengths=None):
        text_output = self.bert(input_ids=text_encoding[0],
                                attention_mask=text_encoding[1],
                                output_attentions=True,
                                output_hidden_states=True).last_hidden_state

        x_global = self.global_pooling_layer(text_output.permute(0, 2, 1)).squeeze(-1)

        x_local, scores = self.local_pooling_layer(text_output[:, 1:, :])
        x_local = self.local_fc(x_local)

        return x_global, x_local


class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.global_pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.local_pooling_layer = SelectivePoolingLayer(in_channels=768, out_channels=512, k=8)
        self.local_fc = nn.Sequential(nn.LayerNorm(512),
                                      nn.Linear(in_features=512, out_features=512))

    def forward(self, img_encoding):
        img_output = self.vit(pixel_values=img_encoding,
                              output_hidden_states=True).last_hidden_state

        # x_global = img_output[:, 0, :]
        x_global = self.global_pooling_layer(img_output.permute(0, 2, 1)).squeeze(-1)
        x_local, scores = self.local_pooling_layer(img_output[:, 1:, :])
        x_local = self.local_fc(x_local)

        return x_global, x_local


class MeanPoolingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MeanPoolingLayer, self).__init__()
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, lengths=None):
        if lengths is not None:
            mean_weights = (1 / lengths).unsqueeze(1).to(x.device)
            padding_mask = (
                    torch.arange(x.shape[1], device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)).to(
                x.device)
            outputs = self.fc((x * ~padding_mask.unsqueeze(2)).sum(dim=1) * mean_weights)
        else:
            outputs = self.fc(x.mean(dim=1))

        return outputs


class PoolingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PoolingLayer, self).__init__()
        self.attn_layer = nn.Linear(in_features=in_channels, out_features=1)
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, lengths=None):
        weights = torch.sigmoid(self.attn_layer(x))
        if lengths is not None:
            mean_weights = (1 / lengths).unsqueeze(1).to(x.device)
            padding_mask = (
                    torch.arange(x.shape[1], device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)).to(
                x.device)
            outputs = self.fc((x * weights * ~padding_mask.unsqueeze(2)).sum(dim=1) * mean_weights)
        else:
            outputs = self.fc((x * weights).mean(dim=1))

        return outputs


class SelectivePoolingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(SelectivePoolingLayer, self).__init__()
        self.k = k
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, dropout=0.1)
        self.scorer = nn.Sequential(nn.Linear(in_features=in_channels, out_features=1),
                                    nn.Sigmoid())
        self.topk = PerturbedTopK(k=k)
        # self.fc = nn.Linear(in_features=in_channels * k, out_features=out_channels)
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        attn = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))[0].transpose(0, 1)
        scores = self.scorer(attn).squeeze(-1)
        # print(torch.topk(scores, k=4, dim=1)[1])
        topk_weights = self.topk(scores)
        topk_feats = (topk_weights.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=2).mean(dim=1)

        # non-differentiable top-k
        # _, ids = torch.topk(scores, dim=1, k = self.k)
        # random selection
        # ids = torch.randint(high=scores.shape[-1], size=(scores.shape[0], self.k), device=scores.device)
        # topk = F.one_hot(ids, num_classes=scores.shape[-1])
        # topk_feats = (x.unsqueeze(1) * topk.unsqueeze(-1)).sum(2)

        # outputs = self.fc(topk_feats.reshape(x.shape[0], -1))

        outputs = self.fc(topk_feats)

        return outputs, scores


class AttentionPoolingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionPoolingLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, dropout=0.1)
        self.fc1 = nn.Linear(in_features=in_channels, out_features=1)
        self.fc2 = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        attn = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))[0].transpose(0, 1)
        weights = F.softmax(self.fc1(attn).squeeze(-1), dim=-1).unsqueeze(-1)
        output = self.fc2((weights * x).sum(dim=1))
        return output, weights


class DualTransformers(nn.Module):
    def __init__(self, margin=0.3, margin2=0.95, dataset='PS', temp=1.0, train_classes=None):
        super(DualTransformers, self).__init__()

        # bert_config = BertConfig()
        # vit_config = ViTConfig()
        # self.vit = ViTModel(vit_config)
        # self.bert = BertModel(bert_config)
        # self.dataset = dataset

        # self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.train_classes = train_classes
        self.IE = ImageEncoder()
        # self.IE = ViTEncoder()
        self.TE = TextEncoder()
        # self.TE = BertEncoder()

        # self.img_pooling_layer = PoolingLayer(in_channels=512, out_channels=512)
        # self.text_pooling_layer = PoolingLayer(in_channels=512, out_channels=512)

        # self.img_global_d = GlobalDiscriminator(global_channels=512, local_channels=512)
        # self.text_global_d = GlobalDiscriminator(global_channels=512, local_channels=512)
        # self.img_local_d = LocalDiscriminator(global_channels=512, local_channels=512)
        # self.text_local_d = LocalDiscriminator(global_channels=512, local_channels=512)

        self.ae = AE(img_channels=512, text_channels=512)
        # self.ae = AE(img_channels=768, text_channels=256)
        # self.ae = LEAE(img_channels=512, text_channels=256, train_classes=self.train_classes)
        # self.ae = VanillaAE(img_channels=768, text_channels=768)

        # self.img_atten_layer = Linear(in_features=768, out_features=1)
        # self.text_atten_layer = Linear(in_features=768, out_features=1)
        # self.img_fc = nn.Linear(in_features=768, out_features=512)
        # self.text_fc = nn.Linear(in_features=768, out_features=512)

        # self.img_kd = ResNetKD()
        # self.img_vae = VanillaVAE(in_channels=768, latent_dim=256, hidden_dim=256)
        # self.text_vae = VanillaVAE(in_channels=768, latent_dim=256, hidden_dim=256)
        # self.img_kd = ViTKD()
        # self.img_kd = GTLabel()
        # self.GAN = VanillaGAN(in_features=768, out_features=512)

        # if self.dataset == 'PS':
        #     logger.info("Loaded Pascal Sentence word embeddings...")
        #     self.trn_labels_emb = torch.FloatTensor(np.load("./w2v/ps_trn_embs.npy"))
        # self.tst_labels_emb = torch.FloatTensor(np.load("./w2v/ps_tst_embs.npy"))
        # elif self.dataset == 'WP_MM':
        #     logger.info("Loaded  Wikipedia word embeddings...")
        #     self.trn_labels_emb = torch.FloatTensor(np.load("./w2v/wp_trn_embs.npy"))
        # self.tst_labels_emb = torch.FloatTensor(np.load("./w2v/wp_tst_embs.npy"))
        # elif self.dataset == 'XM':
        #     logger.info("Loaded XMediaNet word embeddings...")
        #     self.trn_labels_emb = torch.FloatTensor(np.load("./w2v/xm_trn_embs.npy"))
        # self.tst_labels_emb = torch.FloatTensor(np.load("./w2v/xm_tst_embs.npy"))

        # self.img_kd = Proj_Block(in_features=768, out_features=300)
        # for p in self.bert.parameters():
        #     if p.dim() > 1:
        #         xavier_uniform_(p)
        # for p in self.ae.parameters():
        #     if p.dim() > 1:
        #         xavier_uniform_(p)

    def forward(self, img_encoding, text_encoding):
        # img_output = self.vit(pixel_values=img_encoding,
        #                       output_attentions=True,
        #                       output_hidden_states=True)
        #
        # text_output = self.bert(input_ids=text_encoding[0],
        #                         attention_mask=text_encoding[1],
        #                         output_attentions=True,
        #                         output_hidden_states=True)

        # img_last_hidden_state = img_output.last_hidden_state
        # text_last_hidden_state = text_output.last_hidden_state
        # img_emb = self.fuse_features(img_last_hidden_state, type="img")
        # text_emb = self.fuse_features(text_last_hidden_state, type="text")
        # img_cls_emb = img_last_hidden_state[:, 0, :]
        # text_cls_emb = text_last_hidden_state[:, 0, :]

        img_global, img_local = self.IE(img_encoding)
        # img_emb = self.fuse_features(img_local, type='img')

        text_global, text_local = self.TE(text_encoding)

        # text_emb = self.fuse_features(text_local, type='text')

        # img_emb = self.img_pooling_layer(img_local)
        # text_emb = self.text_pooling_layer(text_local, text_lengths)

        img_ae_emb, text_ae_emb, ae_loss = self.ae(img_encoding, img_global, text_global)
        # img_ae_emb, text_ae_emb, ae_loss = self.ae(img_global, text_global, labels)
        # with open('img_features.txt', 'a') as f:
        #     s = img_local.cpu().numpy().tolist()
        #     for score in s:
        #         f.write(str(score) + '\n')
        # with open('text_features.txt', 'a') as f:
        #     s = text_local.cpu().numpy().tolist()
        #     for score in s:
        #         f.write(str(score) + '\n')

        # img_emb_prime = torch.cat((img_emb[1:], img_emb[0].unsqueeze(0)), dim=0)
        # text_emb_prime = torch.cat((text_emb[1:], text_emb[0].unsqueeze(0)), dim=0)
        # Ej = -F.softplus(-self.img_global_d(img_global, img_emb)).mean() - F.softplus(
        #     -self.text_global_d(text_global, text_emb)).mean()
        # Em = F.softplus(self.img_global_d(img_global, img_emb_prime)).mean() + F.softplus(
        #     self.text_global_d(text_global, text_emb_prime)).mean()
        # GLOBAL = (Em - Ej)
        # img_local_prime = torch.cat((img_local[1:], img_local[0].unsqueeze(0)), dim=0)
        # text_local_prime = torch.cat((text_local[1:], text_local[0].unsqueeze(0)), dim=0)
        # Ej = -F.softplus(-self.img_local_d(img_global, img_local)).mean() - F.softplus(
        #     -self.text_local_d(text_global, text_local)).mean()
        # Em = F.softplus(self.img_local_d(img_global, img_local_prime)).mean() + F.softplus(
        #     self.text_local_d(text_global, text_local_prime)).mean()
        # LOCAL = (Em - Ej)

        # img_emb = self.img_fc(torch.mean(img_last_hidden_state[:, 1:, :], dim=1))
        # text_emb = self.text_fc(torch.mean(text_last_hidden_state[:, 1:-1, :], dim=1))
        # img_emb = self.img_fc(img_cls_emb)
        # text_emb = self.text_fc(text_cls_emb)

        # img_cls_preds, text_cls_preds = self.img_kd(img_cls_emb, text_cls_emb)
        # cls_preds = None

        # img_cls_emb, img_embeddings = self.vit(img_encoding)
        # img_emb = self.img_fc(torch.mean(img_embeddings, dim=1))
        # img_emb = self.fuse_features(img_embeddings, type="img_res")
        # text_cls_emb, text_embeddings = self.bert(text_encoding)
        # text_emb = self.text_fc(torch.mean(text_embeddings, dim=1))
        # text_emb = self.fuse_features(text_embeddings, type="text_dv")

        # cls_preds, img_cls_preds, text_cls_preds = self.img_kd(img_encoding, img_cls_emb, text_cls_emb)

        # img_vae_args = self.img_vae(img_cls_emb)
        # text_vae_args = self.text_vae(text_cls_emb)
        #
        # img_cls_preds = self.img_vae.encoder(img_cls_emb)
        # text_cls_preds = self.text_vae.encoder(text_cls_emb)
        #
        # img_cls_loss = self.img_vae.loss_function(*img_vae_args)['loss']
        # text_cls_loss = self.text_vae.loss_function(*text_vae_args)['loss']
        # vae_loss = img_cls_loss + text_cls_loss + ((img_vae_args[0] - text_cls_emb)**2).mean() + ((text_vae_args[0] - img_cls_emb)**2).mean()

        # one-hot imagenet labels
        # cls_preds = torch.max(cls_preds, dim=1)[1]

        # img_cls_preds = torch.matmul(self.img_kd(img_cls_emb), self.trn_labels_emb.to(img_emb.device).t())
        # text_cls_preds = torch.matmul(self.img_kd(text_cls_emb), self.trn_labels_emb.to(text_emb.device).t())
        # cls_preds = None

        return img_local, text_local, img_ae_emb, text_ae_emb, ae_loss
