import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# --------------------------Main------------------------------- #

class MainNet(nn.Module):

    def __init__(self,channel,sf):
        super(MainNet, self).__init__()
        num_channel = channel
        self.sf=sf
        num_feature = 48
        ####################
        self.T_E = Transformer_E(num_feature)
        self.T_D = Transformer_D(num_feature)
        self.Embedding = nn.Sequential(
            nn.Linear(num_channel+4,num_feature),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(num_feature,num_feature,3,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(num_feature, num_channel, 3, 1, 1)
        )

    def forward(self, HSI, MSI):
        ################LR-HSI###################
        import matplotlib.pyplot as plt

        UP_LRHSI = F.interpolate(HSI,scale_factor=self.sf, mode='bicubic') ### (b N h w)
        UP_LRHSI = UP_LRHSI.clamp_(0,1)
        sz= UP_LRHSI.size(2)
        Data = torch.cat((UP_LRHSI,MSI),1)
        E = rearrange(Data, 'B c H W -> B (H W) c', H = sz)
        E = self.Embedding(E)
        Code = self.T_E(E)
        Highpass = self.T_D(Code)
        Highpass = rearrange(Highpass,'B (H W) C -> B C H W', H = sz)
        Highpass = self.refine(Highpass)
        output = Highpass + UP_LRHSI
        output = output.clamp_(0,1)

        return output


# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer_E(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48, sp_sz=64*64, num_channels = 48,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.pos_embedding = nn.Parameter(torch.randn(1.mat, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer_D(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48 , sp_sz=64*64, num_channels = 48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        for attn1,attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, mask=mask)
            x = ff(x)
        return x
