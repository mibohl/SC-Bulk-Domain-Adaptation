import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class FX(nn.Module):
    def __init__(self, dropout_rate, input_dim, h_dim):
        super(FX, self).__init__()
        self.EnE = torch.nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        output = self.EnE(x)
        return output


class MTLP(nn.Module):  ### drug response predictor ###
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(MTLP, self).__init__()
        self.Sh = nn.Linear(h_dim, z_dim)
        self.bn1 = nn.BatchNorm1d(z_dim)
        self.Drop = nn.Dropout(p=dropout_rate)
        self.Predictor = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        ZX = F.relu(self.Drop(self.bn1(self.Sh(X))))
        yhat = self.Predictor(ZX)
        return yhat


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -1


def grad_reverse(x):
    return GradReverse.apply(x)


class Discriminator(nn.Module):
    def __init__(self, dropout_rate, h_dim):
        super(Discriminator, self).__init__()
        self.D1 = nn.Linear(h_dim, 1)
        self.D1 = torch.nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 1),
        )
        self.Drop1 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = grad_reverse(x)
        yhat = self.Drop1(self.D1(x))
        return torch.sigmoid(yhat)