import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Generator, self).__init__()

        self.model = nn.Sequential(nn.Linear(in_features, 512),
                                     nn.BatchNorm1d(512),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(512, out_features),
                                     nn.Tanh())

    def forward(self, x):
        feat = self.model(x)
        return feat


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features, 512),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 1),
                                   nn.Sigmoid())

    def forward(self, x):
        validity = self.model(x).squeeze(1)
        return validity


class VanillaGAN(nn.Module):
    def __init__(self, in_features, out_features):
        super(VanillaGAN, self).__init__()
        self.generator = Generator(in_features, out_features)
        self.discriminator = Discriminator(out_features)

    def forward(self, x):
        feat = self.generator(x)
        validity = self.discriminator(feat)
        return feat, validity

