import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTForImageClassification

import gensim
import gensim.models as g


def JSDLoss(x, y):
    x = F.softmax(x, dim=-1)
    y = F.softmax(y, dim=-1)
    z = 0.5 * (x + y)
    jsd_loss = 0.5 * F.kl_div(z.log(), x, reduction='none').sum(dim=1) + \
               0.5 * F.kl_div(z.log(), y, reduction='none').sum(dim=1)
    return jsd_loss


class AE(nn.Module):
    def __init__(self, img_channels, text_channels):
        super(AE, self).__init__()
        self.frozen_resnet = models.resnet50(pretrained=True)
        for i in self.frozen_resnet.parameters():
            i.requires_grad = False

        # self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        # for i in self.model.parameters():
        #     i.requires_grad = False

        self.img_encoder = nn.Sequential(nn.Linear(img_channels, 512),
                                         nn.LayerNorm(512),
                                         nn.LeakyReLU(),
                                         nn.Linear(512, 1000))
        self.text_encoder = nn.Sequential(nn.Linear(text_channels, 512),
                                          nn.LayerNorm(512),
                                          nn.LeakyReLU(),
                                          nn.Linear(512, 1000))

        self.img_decoder = nn.Sequential(nn.Linear(1000, 512),
                                         nn.LayerNorm(512),
                                         nn.LeakyReLU(),
                                         nn.Linear(512, img_channels))
        self.text_decoder = nn.Sequential(nn.Linear(1000, 512),
                                          nn.LayerNorm(512),
                                          nn.LeakyReLU(),
                                          nn.Linear(512, text_channels))

        self.KL = nn.KLDivLoss(reduction="batchmean")

    def forward(self, encoding, img_cls_emb, text_cls_emb):
        prior = self.frozen_resnet(encoding)
        # prior = self.model(encoding).logits
        # using one-hot label encoding
        # _, ids = torch.max(prior, dim=1)
        # prior = F.one_hot(ids, num_classes=1000).float()
        # prior = torch.randn_like(prior)
        img_latent = self.img_encoder(img_cls_emb)
        text_latent = self.text_encoder(text_cls_emb)
        # img_latent = self.encoder(img_feat)
        # text_latent = self.encoder(text_feat)
        img_recon = self.img_decoder(img_latent)
        text_recon = self.text_decoder(text_latent)

        kl_loss = (self.KL(F.log_softmax(img_latent, dim=-1), F.softmax(prior, dim=-1)) +
                   self.KL(F.log_softmax(text_latent, dim=-1), F.softmax(prior, dim=-1)) +
                   self.KL(F.log_softmax(text_latent, dim=-1), F.softmax(img_latent, dim=-1)) +
                   self.KL(F.log_softmax(img_latent, dim=-1), F.softmax(text_latent, dim=-1))) / 4

        recon_loss = (F.mse_loss(img_recon, img_cls_emb) + F.mse_loss(text_recon, text_cls_emb) +
                      F.mse_loss(img_recon, text_cls_emb) + F.mse_loss(text_recon, img_cls_emb)) / 4
        loss = kl_loss + recon_loss

        return img_latent, text_latent, loss


class LEAE(nn.Module):
    def __init__(self, img_channels, text_channels, train_classes, d2v_model_path='./enwiki_dbow/doc2vec.bin'):
        super(LEAE, self).__init__()
        self.d2v_model = g.Doc2Vec.load(d2v_model_path)
        lbls = []
        for label in train_classes:
            le = self.d2v_model.infer_vector(label)
            lbls.append(le)
        self.le = torch.FloatTensor(lbls)
        self.img_base = nn.Sequential(nn.Linear(img_channels, 512),
                                      nn.LayerNorm(512),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(512, 300))
        self.text_base = nn.Sequential(nn.Linear(text_channels, 512),
                                       nn.LayerNorm(512),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Linear(512, 300))

        self.KL = nn.KLDivLoss(reduction="batchmean")

    def forward(self, img_cls_emb, text_cls_emb, labels=None):
        img_latent = self.img_base(img_cls_emb)
        text_latent = self.text_base(text_cls_emb)

        if labels is not None:
            embeddings = self.le[labels].to(img_cls_emb.device)
            kl_loss = (self.KL(F.log_softmax(img_latent, dim=-1), F.softmax(embeddings, dim=-1)) + \
                       self.KL(F.log_softmax(text_latent, dim=-1), F.softmax(embeddings, dim=-1))) / 2
        # recon_loss = (F.mse_loss(img_recon, img_feat) + F.mse_loss(text_recon, text_feat)) / 2
        else:
            kl_loss = 0

        loss = kl_loss
        return img_latent, text_latent, loss
