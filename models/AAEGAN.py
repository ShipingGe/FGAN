import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import grad as torch_grad
from .contrastive_loss import MMDLoss


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(input_dim, hidden_dim),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(hidden_dim, hidden_dim))

        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        hidden_states = self.encoder(x)
        outputs = self.decoder(hidden_states)
        return hidden_states, outputs


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(input_dim // 2, 1))

    def forward(self, x, label_encoding):
        h = torch.cat([x, label_encoding], dim=1)
        validity = self.model(h)
        return validity


class AAEGAN(nn.Module):
    def __init__(self):
        super(AAEGAN, self).__init__()
        # self.vgg = nn.Sequential(*list(models.vgg19(pretrained=True).children())[:2])
        # self.vgg_head = nn.Sequential(list(models.vgg19(pretrained=True).children())[2][:-3])

        self.img_ae = AutoEncoder(input_dim=768, hidden_dim=512)
        self.text_ae = AutoEncoder(input_dim=768, hidden_dim=512)
        self.label_ae = AutoEncoder(input_dim=300, hidden_dim=512)

        self.img_disc = Discriminator(input_dim=768 + 300)
        self.text_disc = Discriminator(input_dim=768 + 300)

        # self.MMD = MMDLoss()

        # for i in self.vgg.parameters():
        #     i.requires_grad = False
        # for i in self.vgg_head.parameters():
        #     i.requires_grad = False

    def forward(self, imgs, texts, labels):
        # img_aftervgg = self.vgg_head(self.vgg(imgs).reshape(imgs.shape[0], -1))

        img_hidden, img_decoded = self.img_ae(imgs)
        text_hidden, text_decoded = self.text_ae(texts)
        label_hidden, label_decoded = self.label_ae(labels)

        # recon_loss = F.mse_loss(img_decoded, img_aftervgg) + F.mse_loss(text_decoded, texts) + F.mse_loss(label_decoded, labels)

        # mmd_loss = self.MMD(label_hidden, img_hidden, text_hidden)

        # cross_modality_loss = F.mse_loss(img_hidden, text_hidden) + F.mse_loss(img_hidden, label_hidden) + F.mse_loss(text_hidden, label_hidden)

        # da_loss = mmd_loss + cross_modality_loss

        # return img_hidden, text_hidden, img_decoded, text_decoded, recon_loss, da_loss, img_aftervgg
        return img_hidden, text_hidden, label_hidden, img_decoded, text_decoded, label_decoded
