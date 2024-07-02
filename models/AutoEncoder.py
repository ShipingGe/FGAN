import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTForImageClassification

import gensim
import gensim.models as g
import numpy as np



class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar, z


class FGLA(nn.Module):
    def __init__(self, img_dim, text_dim, latent_dim):
        super(FGLA, self).__init__()

        self.img_vae = VAE(img_dim, latent_dim, latent_dim)

        self.text_vae = VAE(text_dim, latent_dim, latent_dim)

        self.img_emb_layer = nn.Linear(latent_dim, 1000)
        self.text_emb_layer = nn.Linear(latent_dim, 1000)

        self.frozen_resnet = models.resnet50(weights="IMAGENET1K_V1")
        for i in self.frozen_resnet.parameters():
            i.requires_grad = False

        self.KL = nn.KLDivLoss(reduction="batchmean")

    def forward(self, img_encoding, image_features, text_features):
        prior = self.frozen_resnet(img_encoding)

        images_recon, images_mu, images_logvar, img_z = self.img_vae(image_features)

        texts_recon, texts_mu, texts_logvar, text_z = self.text_vae(text_features)

        image_loss = vae_loss_func(image_features, images_recon, images_mu, images_logvar)
        text_loss = vae_loss_func(text_features, texts_recon, texts_mu, texts_logvar)
        vae_loss = image_loss + text_loss

        img_sementic_pred = self.img_emb_layer(images_mu)
        text_semantic_pred = self.text_emb_layer(texts_mu)

        align_loss = (self.KL(F.log_softmax(img_sementic_pred, dim=-1), F.softmax(prior, dim=-1)) +
                      self.KL(F.log_softmax(text_semantic_pred, dim=-1), F.softmax(prior, dim=-1)) +
                      self.KL(F.log_softmax(text_semantic_pred, dim=-1), F.softmax(img_sementic_pred, dim=-1)) +
                      self.KL(F.log_softmax(img_sementic_pred, dim=-1), F.softmax(text_semantic_pred, dim=-1))) / 4

        loss = vae_loss + align_loss

        return img_sementic_pred, text_semantic_pred, loss


# 定义损失函数
def vae_loss_func(x, x_recon, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    # print(kl_loss)
    return recon_loss + 1e-4 * kl_loss
    # return recon_loss


def compute_mmd(x, y, kernel='rbf', bandwidth=1.0):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    if kernel == 'rbf':
        K = torch.exp(-(rx.t() + rx - 2 * xx) / (2 * bandwidth ** 2))
        L = torch.exp(-(ry.t() + ry - 2 * yy) / (2 * bandwidth ** 2))
        M = torch.exp(-(rx.t() + ry - 2 * zz) / (2 * bandwidth ** 2))
    elif kernel == 'linear':
        K = xx
        L = yy
        M = zz
    else:
        raise ValueError('Invalid kernel type')

    mmd = K.mean() + L.mean() - 2 * M.mean()
    return mmd
