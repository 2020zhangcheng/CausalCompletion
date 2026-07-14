import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradientReverse.apply(x, lambd)

class GlobalDiscriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    def forward(self,z):
        # z B C
        return self.net(z)

class LocalDiscriminator(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim,256,1),
            nn.ReLU(),
            nn.Conv1d(256,1,1)
        )

    def forward(self,z):
        # B M C
        z=z.transpose(1,2)
        out=self.net(z)
        return out.transpose(1,2)
