import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=opt.lr_decay_gamma,
            patience=opt.lr_decay_patience,
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.lr_policy
        )

    return scheduler


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "mean_space":
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == "mean_channel":
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)

    return net

class SpectralAttentionBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=1):
        super(SpectralAttentionBlock, self).__init__()
        
        self.num_heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(dim, dim_head*heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head*heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head*heads, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        

    def forward(self, x_in):
        b,c,h,w = x_in.shape
        x = x_in.permute(0,2,3,1).reshape(b,w*h,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp),
        )
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c).permute(0,3,1,2)
        return out_c, attn


class FeadForardNetwork(nn.Module):
    def __init__(self, in_c, mult=4):
        super(FeadForardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,in_c*mult,1,1,0,bias=False),
            nn.GELU(),
            nn.Conv2d(in_c*mult,in_c*mult,3,1,1,bias=False,groups=in_c*mult),
            nn.GELU(),
            nn.Conv2d(in_c*mult,in_c,1,1,0,bias=False)
        )
    
    def forward(self, x):
        return self.net(x)
        

class SpectralTransformer(nn.Module):
    def __init__(self, in_c, dim, dim_head=40, heads=1):
        super(SpectralTransformer, self).__init__()
        self.sab = SpectralAttentionBlock(dim, dim_head, heads)
        self.ffn = FeadForardNetwork(in_c)
        self.norm = nn.LayerNorm(in_c)
        

    def forward(self, x):
        x_in = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x_in, attn = self.sab(x_in)
        x_in = x + x_in
        x_out = self.norm(x_in.permute(0,2,3,1)).permute(0,3,1,2)
        x_out = self.ffn(x_out)
        x_out = x_out + x_in
        return x_out, attn
    

class CFormer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = CRATE(dim, depth=2, heads=3, dim_head=16, drop_out=0., ista=0.1)
        self.Embedding = nn.Sequential(
            nn.Linear(dim, dim),
        )
    
    def forward(self, x):
        sz = x.size(2)
        E = rearrange(x, "B c H W -> B (H W) c", H=sz)
        E = self.Embedding(E)
        Code = self.encoder(E)
        Code = rearrange(Code, 'B (H W) C -> B C H W', H = sz)
        return Code

class CRATE(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, drop_out=0., ista=0.1):
        super(CRATE, self).__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ATtention(dim, heads=heads, dim_head=dim_head, dropout=drop_out)),
                PreNorm(dim, FEedForward(dim, dim, dropout=drop_out, step_size=ista))
            ]))
    
    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            grad_x = attn(x) + x
            x = ff(grad_x) + grad_x
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FEedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output

class ATtention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(ATtention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def define_feature_concat(
    in_c, scale, psf, H, Hc, W, Wc, gpu_ids, init_type="kaiming", init_gain=0.02
):
    net = FeatureConcat(in_c, scale, psf, H, Hc, W, Wc, )
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class FeatureConcat(nn.Module):
    def __init__(self, in_c, scale, psf, H, Hc, W, Wc):
        super(FeatureConcat, self).__init__()
        self.conv_concat = nn.Sequential(
            nn.Conv2d(in_c*2, in_c, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_c, in_c, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_c, in_c, 1, 1, 0, bias=False),
            nn.LeakyReLU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c, scale, scale),
            nn.LeakyReLU(),
        )
        self.transfer_H_NH = nn.Sequential(
            nn.Conv2d(in_channels=H, out_channels=Hc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        self.transfer_W_NW = nn.Sequential(
            nn.Conv2d(in_channels=W, out_channels=Wc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        self.psf = psf

    def forward(self, ct1, ct2):
        upsample_feature_1 = torch.cat([self.psf(ct1), ct2], dim=1)
        upsample_feature_1 = self.conv_concat(upsample_feature_1)
        upsample_feature_1 = self.upsample(upsample_feature_1)

        upsample_feature_1 = upsample_feature_1.permute(0, 2, 1, 3)
        upsample_feature_1 = self.transfer_H_NH(upsample_feature_1)
        upsample_feature_1 = upsample_feature_1.permute(0, 2, 1, 3)

        # transfer W to NW
        upsample_feature_1 = upsample_feature_1.permute(0, 3, 2, 1)
        upsample_feature_1 = self.transfer_W_NW(upsample_feature_1)
        upsample_feature_1 = upsample_feature_1.permute(0, 3, 2, 1)
        return upsample_feature_1
        

def define_feature_unet(
    out_channel, scale, H, Hc, W, Wc, gpu_ids, init_type="kaiming", init_gain=0.02
):
    net = FeatureUNet(out_channel, scale, H, Hc, W, Wc)
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)



def define_hrmsi_feature(
    in_channel, out_channel, gpu_ids, init_type="kaiming", init_gain=0.02
):
    net = GenerateFeatureHrMSI(in_channel=in_channel, out_channel=out_channel)
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class GenerateFeatureHrMSI(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GenerateFeatureHrMSI, self).__init__()

        self.feature_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64 * 2, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64 * 2,
                out_channels=64 * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64 * 4,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        msi_feature = self.feature_embedding(x)
        return msi_feature


def define_lrhsi_feature(
    in_channel, out_channel, gpu_ids, init_type="kaiming", init_gain=0.02
):
    net = GenerateFeatureLrHSI(in_channel=in_channel, out_channel=out_channel)
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class GenerateFeatureLrHSI(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GenerateFeatureLrHSI, self).__init__()
        self.feature_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64 * 2, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64 * 2,
                out_channels=64 * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64 * 4,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        lr_hsi_feature = self.feature_embedding(x)
        return lr_hsi_feature


def define_lrdict_wh(
    code_scale, hsi_scale_w, hsi_scale_h, gpu_ids, init_type="normal", init_gain=0.02
):
    net = LrHSIDictionaryWH(
        code_scale=code_scale, hsi_scale_w=hsi_scale_w, hsi_scale_h=hsi_scale_h
    )
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class LrHSIDictionaryWH(nn.Module):
    def __init__(self, code_scale, hsi_scale_w, hsi_scale_h):
        # 字典的更新，用卷积层代替
        super(LrHSIDictionaryWH, self).__init__()
        self.conv_w = nn.Conv2d(code_scale[0], hsi_scale_w, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_h = nn.Conv2d(code_scale[1], hsi_scale_h, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        nx = x.permute(0, 2, 1, 3)
        nx = self.conv_w(nx)
        nx = nx.permute(0, 2, 1, 3)

        nx = nx.permute(0, 3, 2, 1)
        nx = self.conv_h(nx)
        nx = nx.permute(0, 3, 2, 1)
        return nx


def define_lrdict_wht(
    code_scale, hsi_scale_w, hsi_scale_h, gpu_ids, init_type="normal", init_gain=0.02
):
    net = LrHSIDictionaryWHT(
        code_scale=code_scale, hsi_scale_w=hsi_scale_w, hsi_scale_h=hsi_scale_h
    )
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)



class LrHSIDictionaryWHT(nn.Module):
    def __init__(self, code_scale, hsi_scale_w, hsi_scale_h):
        # 字典的更新，用卷积层代替
        super(LrHSIDictionaryWHT, self).__init__()
        self.conv_w = nn.Conv2d(hsi_scale_w, code_scale[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_h = nn.Conv2d(hsi_scale_h, code_scale[1], kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        nx = x.permute(0, 2, 1, 3)
        nx = self.conv_w(nx)
        nx = nx.permute(0, 2, 1, 3)

        nx = nx.permute(0, 3, 2, 1)
        nx = self.conv_h(nx)
        nx = nx.permute(0, 3, 2, 1)
        return nx



def define_lrdict_s(
    code_scale, hsi_scale_s, gpu_ids, init_type="normal", init_gain=0.02
):
    net = LrHSIDictionaryS(code_scale=code_scale, hsi_scale_s=hsi_scale_s)
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class LrHSIDictionaryS(nn.Module):
    def __init__(self, code_scale, hsi_scale_s):
        # 字典的更新，用卷积层代替
        super(LrHSIDictionaryS, self).__init__()

        self.conv_s = nn.Conv2d(code_scale[2], hsi_scale_s, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        return self.conv_s(x).clamp_(0, 1)


def define_lrdict_st(
    code_scale, hsi_scale_s, gpu_ids, init_type="normal", init_gain=0.02
):
    net = LrHSIDictionaryST(code_scale=code_scale, hsi_scale_s=hsi_scale_s)
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class LrHSIDictionaryST(nn.Module):
    def __init__(self, code_scale, hsi_scale_s):
        # 字典的更新，用卷积层代替
        super(LrHSIDictionaryST, self).__init__()

        self.conv_s = nn.Conv2d(hsi_scale_s, code_scale[2], kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        return self.conv_s(x)


def define_hrdict_wh(
    code_scale, msi_scale_w, msi_scale_h, gpu_ids, init_type="normal", init_gain=0.02
):
    net = HrMSIDictionaryWH(
        code_scale=code_scale, msi_scale_w=msi_scale_w, msi_scale_h=msi_scale_h
    )
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class HrMSIDictionaryWH(nn.Module):
    def __init__(self, code_scale, msi_scale_w, msi_scale_h):
        # 字典的更新，用卷积层代替
        super(HrMSIDictionaryWH, self).__init__()
        self.conv_w = nn.Conv2d(code_scale[0], msi_scale_w, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_h = nn.Conv2d(code_scale[1], msi_scale_h, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        nx = x.permute(0, 2, 1, 3)
        nx = self.conv_w(nx)
        nx = nx.permute(0, 2, 1, 3)

        nx = nx.permute(0, 3, 2, 1)
        nx = self.conv_h(nx)
        nx = nx.permute(0, 3, 2, 1)
        return nx


def define_hrdict_wht(
    code_scale, msi_scale_w, msi_scale_h, gpu_ids, init_type="normal", init_gain=0.02
):
    net = HrMSIDictionaryWHT(
        code_scale=code_scale, msi_scale_w=msi_scale_w, msi_scale_h=msi_scale_h
    )
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class HrMSIDictionaryWHT(nn.Module):
    def __init__(self, code_scale, msi_scale_w, msi_scale_h):
        # 字典的更新，用卷积层代替
        super(HrMSIDictionaryWHT, self).__init__()
        self.conv_w = nn.Conv2d(msi_scale_w, code_scale[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_h = nn.Conv2d(msi_scale_h, code_scale[1], kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        nx = x.permute(0, 2, 1, 3)
        nx = self.conv_w(nx)
        nx = nx.permute(0, 2, 1, 3)

        nx = nx.permute(0, 3, 2, 1)
        nx = self.conv_h(nx)
        nx = nx.permute(0, 3, 2, 1)
        return nx


def define_hrdict_s(
    code_scale, msi_scale_s, gpu_ids, init_type="normal", init_gain=0.02
):
    net = HrMSIDictionaryS(code_scale=code_scale, msi_scale_s=msi_scale_s)
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class HrMSIDictionaryS(nn.Module):
    def __init__(self, code_scale, msi_scale_s):
        # 字典的更新，用卷积层代替
        super(HrMSIDictionaryS, self).__init__()
        self.conv_s = nn.Conv2d(
            code_scale[2], msi_scale_s, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        return self.conv_s(x).clamp_(0, 1)
    

def define_hrdict_st(
    code_scale, msi_scale_s, gpu_ids, init_type="normal", init_gain=0.02
):
    net = HrMSIDictionaryST(code_scale=code_scale, msi_scale_s=msi_scale_s)
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class HrMSIDictionaryST(nn.Module):
    def __init__(self, code_scale, msi_scale_s):
        # 字典的更新，用卷积层代替
        super(HrMSIDictionaryST, self).__init__()
        self.conv_s = nn.Conv2d(
            msi_scale_s, code_scale[2], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # batch, channel, height, weight = list(x.size())
        return self.conv_s(x)
        

def define_psf(scale, gpu_ids, init_type="mean_space", init_gain=0.02):
    net = PSF(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)


class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.scale = scale
        self.conv2d = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixelshuffle = nn.PixelUnshuffle(scale)
        self.net = nn.Conv2d(scale * scale, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        features = torch.cat(
            [
                self.conv2d(x[:, i, :, :].view(batch, 1, height, weight))
                for i in range(channel)
            ],
            1,
        )
        # features = self.conv2d(features)
        features = self.pixelshuffle(features)
        batch, channel, height, weight = list(features.size())
        features = torch.cat(
            [
                self.net(
                    features[
                        :,
                        i * self.scale * self.scale : (i + 1) * self.scale * self.scale,
                        :,
                        :,
                    ].view(batch, self.scale * self.scale, height, weight)
                )
                for i in range(0, channel // (self.scale * self.scale))
            ],
            1,
        )
        return features  # same as groups=input_c, i.e. channelwise conv


def define_psf_2(scale, gpu_ids, init_type="normal", init_gain=0.02):
    net = PSF2(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)


class PSF2(nn.Module):
    def __init__(self, scale):
        super(PSF2, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)

        P1 = torch.rand(scale, 1) * 1e-4
        P2 = torch.rand(scale, 1) * 1e-4
        self.Conv_P1 = nn.Parameter(P1)
        self.Conv_P2 = nn.Parameter(P2)

    def forward(self, x):
        batch, channel, height, weight = list(x.size()) # [1, C, H, W]

        # torch.reshape()
        
        return torch.cat(
            [
                self.net(x[:, i, :, :].view(batch, 1, height, weight))
                for i in range(channel)
            ],
            1,
        )  # same as groups=input_c, i.e. channelwise conv


def define_hr2msi(args, hsi_channels, msi_channels, sp_matrix, sp_range, gpu_ids, init_type="mean_channel", init_gain=0.02):
    if args.isCal_SRF is False:
        net = matrix_dot_hr2msi(sp_matrix)
    elif args.isCal_SRF is True:
        net = convolution_hr2msi(hsi_channels, msi_channels, sp_range)
    return init_net(net, init_type, init_gain, gpu_ids)


class convolution_hr2msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(convolution_hr2msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:, 1] - self.sp_range[:, 0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()
        self.msi_channels = msi_channels

        A = torch.ones(hsi_channels, msi_channels) * 5e-1 + torch.rand(hsi_channels, msi_channels) * 5e-2

        self.srf_matrix = nn.Parameter(A)

        # old version
        self.conv2d_list = nn.ModuleList(
            [nn.Conv2d(x, 1, 1, 1, 0, bias=False) for x in self.length_of_each_band]
        )

        # self.conv2d_list = nn.ModuleList([
        #     nn.Conv2d(in_channels=hsi_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(msi_channels)
        # ])

    def forward(self, input):
        # b,c,h,w = input.size()
        # output = torch.matmul(torch.reshape(input.permute(0,2,3,1), [b,h*w,c]), self.srf_matrix)
        # output = torch.reshape(output,[b,h,w,self.msi_channels]).permute(0,3,1,2)
        # return output.clamp_(0,1)
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[
                :, self.sp_range[i, 0] : self.sp_range[i, 1] + 1, :, :
            ]
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            # out = layer(input).div_(layer.weight.data.max(dim=1).view(1)) # normalize
            cat_list.append(out)
        return torch.cat(cat_list, 1).clamp_(0, 1)


class matrix_dot_hr2msi(nn.Module):
    def __init__(self, spectral_response_matrix):
        super(matrix_dot_hr2msi, self).__init__()
        self.register_buffer(
            "sp_matrix", torch.tensor(spectral_response_matrix.transpose(1, 0)).float()
        )
        A = torch.rand(103, 4) * 5e-2 # torch.ones(103, 4) * 5e-1 + torch.rand(103, 4) * 5e-2
        self.srf_matrix = nn.Parameter(A)

    def __call__(self, x):
        batch, channel_hsi, heigth, width = list(x.size())
        channel_msi_sp, channel_hsi_sp = list(self.sp_matrix.size())
        hmsi = torch.bmm(
            self.sp_matrix.expand(batch, -1, -1),
            torch.reshape(x, (batch, channel_hsi, heigth * width)),
        ).view(batch, channel_msi_sp, heigth, width)
        return hmsi


class NonZeroClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w.clamp_(0, 1e8)


class ZeroOneClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w.clamp_(0, 1)


class SumToOneClipper(object):
    def __call__(self, module):
        if hasattr(module, "weight"):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0, 10)
                w.div_(w.sum(dim=1, keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0, 5)


class FeatureUNet(nn.Module):
    def __init__(self, out_channel, scale, H, Hc, W, Wc):
        super(FeatureUNet, self).__init__()
        self.Wc = Wc
        self.Hc = Hc

        self.iterations = int(np.log2(scale))

        self.ssab1 = SpectralTransformer(in_c=out_channel, dim=out_channel, dim_head=out_channel)
        self.ssab2 = SpectralTransformer(in_c=out_channel, dim=out_channel, dim_head=out_channel)
        self.ssab3 = SpectralTransformer(in_c=out_channel, dim=out_channel, dim_head=out_channel)

        self.crate1 = CFormer(out_channel)
        self.crate2 = CFormer(out_channel)
        self.crate3 = CFormer(out_channel)

        self.msi_crate_1 = CFormer(out_channel)
        self.msi_crate_2 = CFormer(out_channel)
        self.msi_crate_3 = CFormer(out_channel)

        self.conv_channel = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
        )

        self.down_sample_conv_list_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )

        self.down_sample_conv_list_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )

        self.down_sample_conv_list_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )

        self.down_sample_conv_list_4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )

        self.down_sample_conv_list_5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )

        self.dowmsample_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),
        )
        self.dowmsample_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),
        )
        self.dowmsample_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),
        )
        self.dowmsample_4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),
        )
        self.dowmsample_5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),
        )

        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.max_pool_2d = nn.AdaptiveMaxPool2d(1)
        self.fc_1_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_1_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_2_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_2_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_3_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_3_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_4_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_4_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_5_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_5_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_6_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_6_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_7_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_7_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_8_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_8_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_9_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_9_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)
        self.fc_10_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, bias=False)
        self.fc_10_2 = nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, bias=False)

        self.conv_1_3x3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_2_3x3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_3_3x3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_4_3x3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_5_3x3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=scale, mode="bicubic")

        self.conv_channel_hsi = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.lr_hsi_compress_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.lr_hsi_compress_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.lr_hsi_compress_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.lr_hsi_compress_4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.lr_hsi_compress_5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.bottom_layer = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.up_sample_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.up_sample_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.up_sample_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.up_sample_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.up_sample_conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )

        self.skip_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.skip_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.skip_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.skip_up_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.skip_up_conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.skip_connection_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.skip_connection_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.skip_connection_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.skip_connection_4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.skip_connection_5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.transfer_H_NH = nn.Sequential(
            nn.Conv2d(in_channels=H, out_channels=Hc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        self.transfer_W_NW = nn.Sequential(
            nn.Conv2d(in_channels=W, out_channels=Wc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        # self.transfer_CT_spatial = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(W-Wc+1), stride=1, padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(),
        # )
        # self.transfer_CT_spatial = nn.AvgPool2d(kernel_size=(W-Wc+1), stride=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, hr_msi_feature, lr_hsi_feature, transformer=False):
        # lr_hsi_feature_up = self.upsample(lr_hsi_feature)
        # lr_hsi_feature_iter = self.conv_channel_hsi(lr_hsi_feature)
        lr_hsi_feature_iter = lr_hsi_feature

        # feature_cat = torch.cat([lr_hsi_feature_up, hr_msi_feature], dim=1)
        feature_in = self.conv_channel(hr_msi_feature)

        # downsample 1
        feature_downsample_1 = self.relu(feature_in + self.down_sample_conv_list_1(feature_in))
        # feature_downsample_1 = self.relu(feature_in + self.msi_crate_1(feature_in))
        # lr_hsi_feature_iter_1 = self.lr_hsi_compress_1(lr_hsi_feature_iter)
        lr_hsi_feature_iter_1 = lr_hsi_feature_iter
        # lr_hsi_feature_iter_1 = self.crate1(lr_hsi_feature_iter)

        if not transformer:
            # channel attention_1
            avg_channel_1 = self.avg_pool_2d(lr_hsi_feature_iter_1)
            max_channel_1 = self.max_pool_2d(lr_hsi_feature_iter_1)
            avg_out_1 = self.fc_1_2(self.relu(self.fc_1_1(avg_channel_1)))
            max_out_1 = self.fc_1_2(self.relu(self.fc_1_1(max_channel_1)))
            channel_out_1 = avg_out_1 + max_out_1
            channel_out_1 = self.sigmoid(channel_out_1)
            # if (epoch % 100) == 0:
            #     # save the feature map
            #     import scipy.io as sio
            #     save_mat = {"channel_out_1":channel_out_1.data.cpu().float().numpy()}
            #     sio.savemat("/data/hewang/tensor_factorization/CrossAttention/checkpoints/Pavia_scale_8/feature_map/spec_attn1_{}.mat".format(epoch),
            #                 save_mat)
            feature_downsample_1 = feature_downsample_1 * channel_out_1
            lr_hsi_feature_iter_1 = (
                    lr_hsi_feature_iter_1 + lr_hsi_feature_iter_1 * channel_out_1
            )
        else:
            # Transformer to extract channel attention
            lr_hsi_feature_iter_1, spe_attn_1 = self.ssab1(lr_hsi_feature_iter_1)  # spe_attn_1: [C, C]
            _, c, h, w = feature_downsample_1.shape
            feature_downsample_1 = (
                        (feature_downsample_1.squeeze().reshape([c, h * w]).T) @ spe_attn_1.squeeze()).T.reshape(c, h,
                                                                                                                 w).unsqueeze(
                0)

        feature_downsample_1_next_in = self.dowmsample_1(feature_downsample_1)

        # downsample 2
        feature_downsample_2 = self.relu(
            feature_downsample_1_next_in + self.down_sample_conv_list_2(feature_downsample_1_next_in))
        # feature_downsample_2 = self.relu(feature_downsample_1_next_in + self.msi_crate_2(feature_downsample_1_next_in))
        # lr_hsi_feature_iter_2 = self.lr_hsi_compress_2(lr_hsi_feature_iter_1)
        lr_hsi_feature_iter_2 = lr_hsi_feature_iter_1
        # lr_hsi_feature_iter_2 = self.crate2(lr_hsi_feature_iter_1)

        if not transformer:
            # channel attention_2
            avg_channel_2 = self.avg_pool_2d(lr_hsi_feature_iter_2)
            max_channel_2 = self.max_pool_2d(lr_hsi_feature_iter_2)
            avg_out_2 = self.fc_2_2(self.relu(self.fc_2_1(avg_channel_2)))
            max_out_2 = self.fc_2_2(self.relu(self.fc_2_1(max_channel_2)))
            channel_out_2 = avg_out_2 + max_out_2
            channel_out_2 = self.sigmoid(channel_out_2)
            feature_downsample_2 = feature_downsample_2 * channel_out_2
            lr_hsi_feature_iter_2 = (
                    lr_hsi_feature_iter_2 + lr_hsi_feature_iter_2 * channel_out_2
            )
        else:
            # Transformer to extract channel attention
            lr_hsi_feature_iter_2, spe_attn_2 = self.ssab2(lr_hsi_feature_iter_2)  # spe_attn_2: [C, C]
            _, c, h, w = feature_downsample_2.shape
            feature_downsample_2 = ((feature_downsample_2.squeeze().reshape([c, h * w]).T) @ spe_attn_2).T.reshape(c, h,
                                                                                                                   w).unsqueeze(
                0)

        feature_downsample_2_next_in = self.dowmsample_2(feature_downsample_2)

        # downsample 3
        feature_downsample_3 = self.relu(
            feature_downsample_2_next_in + self.down_sample_conv_list_3(feature_downsample_2_next_in))
        # feature_downsample_3 = self.relu(feature_downsample_2_next_in + self.msi_crate_3(feature_downsample_2_next_in))
        # lr_hsi_feature_iter_3 = self.lr_hsi_compress_3(lr_hsi_feature_iter_2)
        # lr_hsi_feature_iter_3 = lr_hsi_feature_iter_2
        lr_hsi_feature_iter_3 = self.crate3(lr_hsi_feature_iter_2)

        if not transformer:
            # channel attention_3
            avg_channel_3 = self.avg_pool_2d(lr_hsi_feature_iter_3)
            max_channel_3 = self.max_pool_2d(lr_hsi_feature_iter_3)
            avg_out_3 = self.fc_3_2(self.relu(self.fc_3_1(avg_channel_3)))
            max_out_3 = self.fc_3_2(self.relu(self.fc_3_1(max_channel_3)))
            channel_out_3 = avg_out_3 + max_out_3
            channel_out_3 = self.sigmoid(channel_out_3)
            feature_downsample_3 = feature_downsample_3 * channel_out_3
            lr_hsi_feature_iter_3 = (
                    lr_hsi_feature_iter_3 + lr_hsi_feature_iter_3 * channel_out_3
            )
        else:
            # Transformer to extract channel attention
            lr_hsi_feature_iter_3, spe_attn_3 = self.ssab3(lr_hsi_feature_iter_3)  # spe_attn_1: [C, C]
            _, c, h, w = feature_downsample_3.shape
            feature_downsample_3 = ((feature_downsample_3.squeeze().reshape([c, h * w]).T) @ spe_attn_3).T.reshape(c, h,
                                                                                                                   w).unsqueeze(
                0)

        feature_downsample_3_next_in = self.dowmsample_3(feature_downsample_3)

        # downsample 4
        feature_downsample_4 = self.relu(
            feature_downsample_3_next_in
            + self.down_sample_conv_list_4(feature_downsample_3_next_in)
        )
        lr_hsi_feature_iter_4 = lr_hsi_feature_iter_3

        if not transformer:
            # channel attention_4
            avg_channel_7 = self.avg_pool_2d(lr_hsi_feature_iter_4)
            max_channel_7 = self.max_pool_2d(lr_hsi_feature_iter_4)
            avg_out_7 = self.fc_7_2(self.relu(self.fc_7_1(avg_channel_7)))
            max_out_7 = self.fc_7_2(self.relu(self.fc_7_1(max_channel_7)))
            channel_out_7 = avg_out_7 + max_out_7
            channel_out_7 = self.sigmoid(channel_out_7)
            feature_downsample_4 = feature_downsample_4 * channel_out_7
            lr_hsi_feature_iter_4 = (
                lr_hsi_feature_iter_4 + lr_hsi_feature_iter_4 * channel_out_7
            )

        feature_downsample_4_next_in = self.dowmsample_4(feature_downsample_4)

        # downsample 5
        feature_downsample_5 = self.relu(
            feature_downsample_4_next_in
            + self.down_sample_conv_list_5(feature_downsample_4_next_in)
        )
        lr_hsi_feature_iter_5 = lr_hsi_feature_iter_4

        if not transformer:
            # channel attention_5
            avg_channel_8 = self.avg_pool_2d(lr_hsi_feature_iter_5)
            max_channel_8 = self.max_pool_2d(lr_hsi_feature_iter_5)
            avg_out_8 = self.fc_8_2(self.relu(self.fc_8_1(avg_channel_8)))
            max_out_8 = self.fc_8_2(self.relu(self.fc_8_1(max_channel_8)))
            channel_out_8 = avg_out_8 + max_out_8
            channel_out_8 = self.sigmoid(channel_out_8)
            feature_downsample_5 = feature_downsample_5 * channel_out_8
            lr_hsi_feature_iter_5 = (
                lr_hsi_feature_iter_5 + lr_hsi_feature_iter_5 * channel_out_8
            )

        feature_downsample_5_next_in = self.dowmsample_5(feature_downsample_5)

        # bottom conv feature
        feature_bottom = self.bottom_layer(
            # upsample_feature_3 = self.bottom_layer(
            torch.cat([feature_downsample_5_next_in, lr_hsi_feature_iter_3], dim=1)
        )
        transformer = True

       # upsample 5
        transformer = True
        # upsample the feature with spatial attention with itself
        upsample_feature_5 = self.up_sample_conv_5(feature_bottom)
        skip_connection_feature_5 = self.skip_connection_5(feature_downsample_5)
        # first spatial attention from skip feature
        spatial_avg_out_5 = torch.mean(skip_connection_feature_5, dim=1, keepdim=True)
        spatial_max_out_5, _ = torch.max(skip_connection_feature_5, dim=1, keepdim=True)
        spatial_out_5 = torch.cat([spatial_avg_out_5, spatial_max_out_5], dim=1)
        spatial_out_5 = self.sigmoid(self.conv_4_3x3(spatial_out_5))
        upsample_feature_5 = upsample_feature_5 * spatial_out_5
        # second spatial attention from itself
        spatial_avg_out_5 = torch.mean(upsample_feature_5, dim=1, keepdim=True)
        spatial_max_out_5, _ = torch.max(upsample_feature_5, dim=1, keepdim=True)
        spatial_out_5 = torch.cat([spatial_avg_out_5, spatial_max_out_5], dim=1)
        spatial_out_5 = self.sigmoid(self.conv_4_3x3(spatial_out_5))
        skip_connection_feature_5 = skip_connection_feature_5 * spatial_out_5
        # spectral attention from lrhsi itself
        if transformer:
            avg_channel_9 = self.avg_pool_2d(upsample_feature_5)
            max_channel_9 = self.max_pool_2d(upsample_feature_5)
            avg_out_9 = self.fc_9_2(self.relu(self.fc_9_1(avg_channel_9)))
            max_out_9 = self.fc_9_2(self.relu(self.fc_9_1(max_channel_9)))
            channel_out_9 = avg_out_9 + max_out_9
            channel_out_9 = self.sigmoid(channel_out_9)
            upsample_feature_5 = upsample_feature_5 * channel_out_9
        # concat the features with spatial attention
        upsample_feature_5 = self.skip_up_conv_4(
            torch.cat([upsample_feature_5, skip_connection_feature_5], dim=1)
        )

        # upsample 4
        transformer = True
        # upsample the feature with spatial attention with itself
        upsample_feature_4 = self.up_sample_conv_4(upsample_feature_5)
        skip_connection_feature_4 = self.skip_connection_4(feature_downsample_4)
        # first spatial attention from skip feature
        spatial_avg_out_4 = torch.mean(skip_connection_feature_4, dim=1, keepdim=True)
        spatial_max_out_4, _ = torch.max(skip_connection_feature_4, dim=1, keepdim=True)
        spatial_out_4 = torch.cat([spatial_avg_out_4, spatial_max_out_4], dim=1)
        spatial_out_4 = self.sigmoid(self.conv_5_3x3(spatial_out_4))
        upsample_feature_4 = upsample_feature_4 * spatial_out_4
        # second spatial attention from itself
        spatial_avg_out_4 = torch.mean(upsample_feature_4, dim=1, keepdim=True)
        spatial_max_out_4, _ = torch.max(upsample_feature_4, dim=1, keepdim=True)
        spatial_out_4 = torch.cat([spatial_avg_out_4, spatial_max_out_4], dim=1)
        spatial_out_4 = self.sigmoid(self.conv_5_3x3(spatial_out_4))
        skip_connection_feature_4 = skip_connection_feature_4 * spatial_out_4
        # spectral attention from lrhsi itself
        if transformer:
            avg_channel_10 = self.avg_pool_2d(upsample_feature_4)
            max_channel_10 = self.max_pool_2d(upsample_feature_4)
            avg_out_10 = self.fc_10_2(self.relu(self.fc_10_1(avg_channel_10)))
            max_out_10 = self.fc_10_2(self.relu(self.fc_10_1(max_channel_10)))
            channel_out_10 = avg_out_10 + max_out_10
            channel_out_10 = self.sigmoid(channel_out_10)
            upsample_feature_4 = upsample_feature_4 * channel_out_10
        # concat the features with spatial attention
        upsample_feature_4 = self.skip_up_conv_5(
            torch.cat([upsample_feature_4, skip_connection_feature_4], dim=1)
        )

        # upsample 3
        transformer = True
        # upsample the feature with spatial attention with itself
        upsample_feature_3 = self.up_sample_conv_3(upsample_feature_4)
        # upsample_feature_3 = self.up_sample_conv_3(feature_bottom)
        skip_connection_feature_3 = self.skip_connection_3(feature_downsample_3)
        # first spatial attention from skip feature
        spatial_avg_out_3 = torch.mean(skip_connection_feature_3, dim=1, keepdim=True)
        spatial_max_out_3, _ = torch.max(skip_connection_feature_3, dim=1, keepdim=True)
        spatial_out_3 = torch.cat([spatial_avg_out_3, spatial_max_out_3], dim=1)
        spatial_out_3 = self.sigmoid(self.conv_1_3x3(spatial_out_3))
        upsample_feature_3 = upsample_feature_3 * spatial_out_3
        # second spatial attention from itself
        spatial_avg_out_3 = torch.mean(upsample_feature_3, dim=1, keepdim=True)
        spatial_max_out_3, _ = torch.max(upsample_feature_3, dim=1, keepdim=True)
        spatial_out_3 = torch.cat([spatial_avg_out_3, spatial_max_out_3], dim=1)
        spatial_out_3 = self.sigmoid(self.conv_1_3x3(spatial_out_3))

        skip_connection_feature_3 = skip_connection_feature_3 * spatial_out_3
        # spectral attention from lrhsi itself
        if transformer:
            avg_channel_4 = self.avg_pool_2d(upsample_feature_3)
            max_channel_4 = self.max_pool_2d(upsample_feature_3)
            avg_out_4 = self.fc_4_2(self.relu(self.fc_4_1(avg_channel_4)))
            max_out_4 = self.fc_4_2(self.relu(self.fc_4_1(max_channel_4)))
            channel_out_4 = avg_out_4 + max_out_4
            channel_out_4 = self.sigmoid(channel_out_4)
            upsample_feature_3 = upsample_feature_3 * channel_out_4
        else:
            upsample_feature_3 = ((upsample_feature_3.squeeze().reshape([c, h * w]).T) @ spe_attn_3).T.reshape(c, h,
                                                                                                               w).unsqueeze(
                0)
        # concat the features with spatial attention
        upsample_feature_3 = self.skip_up_conv_1(
            torch.cat([upsample_feature_3, skip_connection_feature_3], dim=1)
        )

        # upsample 2
        # upsample the feature with spatial attention with itself
        upsample_feature_2 = self.up_sample_conv_2(upsample_feature_3)
        # upsample_feature_2 = self.up_sample_conv_2(feature_bottom)
        skip_connection_feature_2 = self.skip_connection_2(feature_downsample_2)
        # first spatial attention from skip feature
        spatial_avg_out_2 = torch.mean(skip_connection_feature_2, dim=1, keepdim=True)
        spatial_max_out_2, _ = torch.max(skip_connection_feature_2, dim=1, keepdim=True)
        spatial_out_2 = torch.cat([spatial_avg_out_2, spatial_max_out_2], dim=1)
        spatial_out_2 = self.sigmoid(self.conv_2_3x3(spatial_out_2))
        upsample_feature_2 = upsample_feature_2 * spatial_out_2
        # second spatial attention from itself
        spatial_avg_out_2 = torch.mean(upsample_feature_2, dim=1, keepdim=True)
        spatial_max_out_2, _ = torch.max(upsample_feature_2, dim=1, keepdim=True)
        spatial_out_2 = torch.cat([spatial_avg_out_2, spatial_max_out_2], dim=1)
        spatial_out_2 = self.sigmoid(self.conv_2_3x3(spatial_out_2))

        skip_connection_feature_2 = skip_connection_feature_2 * spatial_out_2
        # spectral attention from lrhsi itself
        if transformer:
            avg_channel_5 = self.avg_pool_2d(upsample_feature_2)
            max_channel_5 = self.max_pool_2d(upsample_feature_2)
            avg_out_5 = self.fc_5_2(self.relu(self.fc_5_1(avg_channel_5)))
            max_out_5 = self.fc_5_2(self.relu(self.fc_5_1(max_channel_5)))
            channel_out_5 = avg_out_5 + max_out_5
            channel_out_5 = self.sigmoid(channel_out_5)
            upsample_feature_2 = upsample_feature_2 * channel_out_5
        else:
            upsample_feature_2 = ((upsample_feature_2.squeeze().reshape([c, h * w]).T) @ spe_attn_2).T.reshape(c, h,
                                                                                                               w).unsqueeze(
                0)
        # concat the features with spatial attention
        upsample_feature_2 = self.skip_up_conv_2(
            torch.cat([upsample_feature_2, skip_connection_feature_2], dim=1)
        )

        # upsample 1
        # upsample the feature with spatial attention with itself
        upsample_feature_1 = self.up_sample_conv_1(upsample_feature_2)
        skip_connection_feature_1 = self.skip_connection_1(feature_downsample_1)
        # first spatial attention from skip feature
        spatial_avg_out_1 = torch.mean(skip_connection_feature_1, dim=1, keepdim=True)
        spatial_max_out_1, _ = torch.max(skip_connection_feature_1, dim=1, keepdim=True)
        spatial_out_1 = torch.cat([spatial_avg_out_1, spatial_max_out_1], dim=1)
        spatial_out_1 = self.sigmoid(self.conv_3_3x3(spatial_out_1))
        upsample_feature_1 = upsample_feature_1 * spatial_out_1
        # second spatial attention from itself
        spatial_avg_out_1 = torch.mean(upsample_feature_1, dim=1, keepdim=True)
        spatial_max_out_1, _ = torch.max(upsample_feature_1, dim=1, keepdim=True)
        spatial_out_1 = torch.cat([spatial_avg_out_1, spatial_max_out_1], dim=1)
        spatial_out_1 = self.sigmoid(self.conv_3_3x3(spatial_out_1))

        skip_connection_feature_1 = skip_connection_feature_1 * spatial_out_1
        # spectral attention from lrhsi itself
        if transformer:
            avg_channel_6 = self.avg_pool_2d(upsample_feature_1)
            max_channel_6 = self.max_pool_2d(upsample_feature_1)
            avg_out_6 = self.fc_6_2(self.relu(self.fc_6_1(avg_channel_6)))
            max_out_6 = self.fc_6_2(self.relu(self.fc_6_1(max_channel_6)))
            channel_out_6 = avg_out_6 + max_out_6
            channel_out_6 = self.sigmoid(channel_out_6)
            upsample_feature_1 = upsample_feature_1 * channel_out_6
        else:
            upsample_feature_1 = ((upsample_feature_1.squeeze().reshape([c, h * w]).T) @ spe_attn_1).T.reshape(c, h,
                                                                                                               w).unsqueeze(
                0)
        # concat the features with spatial attention
        upsample_feature_1 = self.skip_up_conv_3(
            torch.cat([upsample_feature_1, skip_connection_feature_1], dim=1)
        )
        # if (epoch % 100) == 0:
        #     # save the feature map
        #     import scipy.io as sio
        #     save_mat = {"upsample_feature_1":upsample_feature_1.detach().data.cpu().float().numpy()}
        #     sio.savemat(r"D:\\Dataset\\tensor_factorization\\CrossAttention\\checkpoints\\pavia_scale_8\\feature_map_1229\\feature_map1_{}.mat".format(epoch),
        #                 save_mat)

        # upsample_feature_1 = upsample_feature_2

        # Feature mapping to Core-Tensor
        # transfer H to NH
        upsample_feature_1 = upsample_feature_1.permute(0, 2, 1, 3)
 
        upsample_feature_1 = self.transfer_H_NH(upsample_feature_1)
        upsample_feature_1 = upsample_feature_1.permute(0, 2, 1, 3)

        # transfer W to NW
        upsample_feature_1 = upsample_feature_1.permute(0, 3, 2, 1)
        upsample_feature_1 = self.transfer_W_NW(upsample_feature_1)
        upsample_feature_1 = upsample_feature_1.permute(0, 3, 2, 1)

        # upsample_feature_1 = self.transfer_CT_spatial(upsample_feature_1)

        # upsample_feature_1 = torch.resize_as_(upsample_feature_1, torch.zeros([upsample_feature_1.shape[0], upsample_feature_1.shape[1], self.Wc, self.Hc]))s

        # if (epoch % 100) == 0:
        #     # save the feature map
        #     import scipy.io as sio
        #     save_mat = {"core_tensor":upsample_feature_1.data.cpu().float().numpy()}
        #     sio.savemat(r"D:\\Dataset\\tensor_factorization\\CrossAttention\\checkpoints\\pavia_scale_8\\feature_map_1219\\core_tensor_{}.mat".format(epoch),
        #                 save_mat)
        return upsample_feature_1

# 
# network  =define_feature_unet( out_channel=31, H=128, Hc=500, W=128, Wc=500, scale=32,gpu_ids=[0])
# A = torch.rand([1,31,4*32,4*32])
# B = torch.rand([1,31,4,4])
# network(A,B)