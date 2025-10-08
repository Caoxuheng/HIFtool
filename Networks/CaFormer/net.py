import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
import numbers

# ======================Norm Method===========================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# ====================Attention Mechanism==========================

class FeedForward(nn.Module):
    def __init__(self, dim:int, ffn_expansion_factor:float, bias:bool):
        '''

        Args:
            dim: dimension of input feature
            ffn_expansion_factor: expand feature of hidden feature
            bias: is bias
        '''
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, depth):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.temperature1 = nn.Parameter(torch.ones(1, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.test = nn.Identity()
        self.window_size = [8, 8]
        self.depth = depth
        self.shift_size = self.window_size[0] // 2

        self.to_k = nn.Linear(self.window_size[0] ** 2, dim // 4)
        self.to_k2 = nn.Conv2d(in_channels=dim,out_channels=dim//4,kernel_size=1,bias=bias)
        self.to_q_spa = nn.Conv2d(in_channels=dim,out_channels=dim//4,kernel_size=3,padding=1)
        self.to_q_spe = nn.Linear(self.window_size[0]**2,dim//4)
        self.soft = nn.Softmax()

    def forward(self, x):
        if (self.depth) % 2:

            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        b, c, h, w = x.shape

        w_size = self.window_size
        x = rearrange(x, 'b c (h b0) (w b1) -> (b h w) c b0 b1', b0=w_size[0], b1=w_size[1])

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # k b c c
        k = self.to_k( rearrange(k, 'b c b0 b1 -> b c (b0 b1) ')).unsqueeze(-1)
        k = self.to_k2(k).squeeze(-1)
        # q_spatial b c H W
        q_spa = self.to_q_spa(q)
        # q_spectral b C c
        q_spe = self.to_q_spe(rearrange(q, 'b c b0 b1 -> b c (b0 b1) '))



        q_spa = rearrange(q_spa, 'b (head c) h w -> b head c (h w)', head=1)
        q_spe = rearrange(q_spe, 'b (head C) c -> b head C c', head=1)
        k = rearrange(k, 'b (head c1) c -> b head c1 c', head=1)

        q_spa = torch.nn.functional.normalize(q_spa, dim=-1)
        q_spe = torch.nn.functional.normalize(q_spe, dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)
        q_spe_add = q_spe @ k
        attn = (q_spe_add)@ q_spa* self.temperature

        attn = attn.softmax(dim=-1)
        attn = attn.softmax(dim=-2)
        attn = self.test(attn)
        attn = rearrange(attn, 'b head c (h w) -> b (head c) h w', head=1, h=w_size[1], w=w_size[1])
        out = attn * v
        out = rearrange(out, '(b h w) c b0 b1 -> b c (h b0)  (w b1)', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        out = self.project_out(out)
        if (self.depth) % 2:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, depth):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, depth)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    # ===================FFT Fusion Block====================


# ====================FFT==========================
class FFTInteraction_N(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(FFTInteraction_N, self).__init__()
        self.post = nn.Conv2d( in_nc*2, out_nc, 1, 1, 0)
        self.mid = nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups=in_nc)

    def forward(self, x, x_enc, x_dec):

        x_enc = torch.fft.rfft2(x_enc, norm='backward')
        x_dec = torch.fft.rfft2(x_dec, norm='backward')
        x_freq_amp = torch.abs(x_enc)
        x_freq_pha = torch.angle(x_dec)
        x_freq_pha = self.mid(x_freq_pha)
        real = x_freq_amp * torch.cos(x_freq_pha)
        imag = x_freq_amp * torch.sin(x_freq_pha)
        x_recom = torch.complex(real, imag)
        x_recom = torch.fft.irfft2(x_recom)

        out = self.post(torch.cat([x_recom, x], 1))

        return out

class Encoder_block(nn.Module):
    def __init__(self, in_size,  out_size, downsample, use_csff=False, depth=1):
        super( Encoder_block, self).__init__()
        self.downsample = downsample
        self.use_csff = use_csff

        self.block = TransformerBlock(in_size, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',depth = depth)

        if downsample and use_csff:
            self.stage_int = FFTInteraction_N(in_size, in_size)

        if downsample:
            self.downsample = nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = x

        if enc is not None and dec is not None:
            assert self.use_csff

            out = self.stage_int(out, enc, dec)
        out = self.block(out)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down
        else:
            return out

class Decoder_block(nn.Module):
    def __init__(self, in_size, out_size, depth):
        super(Decoder_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv = nn.Conv2d(out_size * 2, out_size, 1, bias=False)
        self.conv_block = Encoder_block(out_size, out_size, False, depth=depth)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.conv(torch.cat([up, bridge], dim=1))
        out = self.conv_block(out)
        return out

class Encoder(nn.Module):
    def __init__(self, n_feat, use_csff=False, depth=4):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()  # []
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(Encoder_block(in_size=n_feat * 2 ** (i),out_size=n_feat * 2 ** (i + 1), downsample=True,
                                           use_csff=use_csff, depth=i))

        self.body.append(
            Encoder_block(in_size=n_feat * 2 ** (depth - 1), out_size=n_feat * 2 ** (depth - 1), downsample=False,
                          use_csff=use_csff, depth=depth))
        self.shift_size = 4

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)

                    x = down(x, encoder_outs[i], decoder_outs[-i - 1])
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)
                    x = down(x)
                else:
                    x = down(x)

        return res, x

class Decoder(nn.Module):
    def __init__(self, n_feat,  depth=4):
        super(Decoder, self).__init__()

        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()  #
        self.shift_size = 4
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(Decoder_block(in_size=n_feat * 2 ** (depth - i - 1), out_size=n_feat * 2 ** (depth - i - 2),
                                         depth=depth - i - 1))

    def forward(self, x, bridges):

        res = []
        for i, up in enumerate(self.body):
            x = up(x, bridges[-i - 1])
            res.append(x)

        return res, x


# ====================Block==========================
class CaBlock(nn.Module):
    def __init__(self,data_fildty_module,in_c, n_feat=80,n_depth=3,):
        super(CaBlock, self).__init__()
        self.shallow_feat = nn.Conv2d(in_c, n_feat, 3, 1, 1, bias=True)
        self.stage_encoder = Encoder(n_feat, use_csff=True, depth=n_depth)
        self.stage_decoder = Decoder(n_feat, depth=n_depth)
        self.data_fildty_module = data_fildty_module



    def forward(self, stage_img, r_0, RGB, f_encoder, f_decoder):

        r_k = self.data_fildty_module(r_0,stage_img,RGB,0)
        b, c, h_inp, w_inp = r_k.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        r_k = F.pad(r_k, [0, pad_w, 0, pad_h], mode='reflect')
        r_k = self.shallow_feat(r_k)
        feat1, f_encoder = self.stage_encoder(r_k, f_encoder, f_decoder)

        f_decoder, last_out = self.stage_decoder(f_encoder, feat1)

        stage_img = last_out + r_k

        return stage_img, feat1, f_decoder


class CaFormer(nn.Module):
    def __init__(self, sf,in_c=31,out_c, n_feat=64, nums_stages=5, n_depth=3):
        super(CaFormer, self).__init__()

        self.conv_tomsi = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_tohsi = torch.nn.Conv2d(in_channels=out_c, out_channels=in_c, kernel_size=3, stride=1, padding=3 // 2)

        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))

        self.delta_3 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_3 = torch.nn.Parameter(torch.tensor(0.9))


        self.conv_downsample = torch.nn.Upsample(scale_factor=1 / sf)
        self.conv_upsample = torch.nn.Upsample(scale_factor=sf)


        self.project_o = nn.Sequential(nn.Conv2d(n_feat,in_c,kernel_size=3,padding=1),nn.ReLU())


        self.scale_factor=sf
        self.body = nn.ModuleList()
        self.nums_stages = nums_stages

        self.shallow_feat = nn.Conv2d(in_c, n_feat, 3, 1, 1, bias=True)
        self.stage_model = nn.ModuleList([CaBlock(data_fildty_module=self.data_fildty_module,in_c=in_c,
            n_feat=n_feat, n_depth=n_depth
        ) for _ in range(self.nums_stages)])

        self.stage_project = nn.ModuleList([nn.Conv2d(in_channels=n_feat,out_channels=in_c,kernel_size=1) for _ in range(self.nums_stages)])

        self.stage1_encoder = Encoder(n_feat, use_csff=False, depth=n_depth)
        self.stage1_decoder = Decoder(n_feat, depth=n_depth)


    def data_fildty_module(self, f_hsi, v, msi, id_layer):
        if id_layer == 0 :
            DELTA = self.delta_0
            ETA = self.eta_0

        elif id_layer == 3:
            DELTA = self.delta_3
            ETA = self.eta_3

        err1 = msi-self.conv_tomsi(f_hsi)
        err1 = self.conv_tohsi(err1)
        err2 = f_hsi-ETA*v
        out = ( 1-DELTA-DELTA*ETA)*f_hsi + DELTA*err1 + DELTA*err2
        return out

    def dun_rc(self, features, recon,lrhsi, msi):
        DELTA = self.delta_0
        ETA = self.eta_0
        down = self.conv_downsample(recon)
        res_lrhsi = self.conv_upsample(down - lrhsi )
        f_msi = self.conv_tomsi(recon)
        res_msi = msi - f_msi
        res_hsi = self.conv_tohsi(res_msi)
        out = (1-DELTA*ETA)*recon +DELTA*res_hsi + DELTA*res_lrhsi + DELTA*ETA*features
        return out

    def forward(self, HSI,MSI):
        output_ = []
        #Stage 1
        b, c,h_inp, w_inp = MSI.shape
        x = torch.nn.functional.interpolate(HSI, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        r_0 = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Stage1
        x = self.shallow_feat(r_0)
        feat1, f_encoder = self.stage1_encoder(x)
        f_decoder, last_out = self.stage1_decoder(f_encoder, feat1)
        stage_img = self.project_o(last_out) + r_0
        output_.append(stage_img)
        # Stage 2_k-1
        for i in range(self.nums_stages):
            stage_img, feat1, f_decoder = self.stage_model[i](stage_img, r_0, MSI ,feat1, f_decoder)
            stage_img = F.leaky_relu(self.stage_project[i](stage_img))
            output_.append(stage_img)
            r_0 = self.dun_rc(stage_img, r_0, HSI, MSI)
        return output_[-1]


if __name__ =='__main__':
    import cv2
    import numpy as np
    x_ = np.random.random([ 128, 128,31])

    f =  CaFormer(32,n_depth=2).cuda()
    srf = np.load('NikonD700.npy')
    HSI = cv2.GaussianBlur(x_,[31,31],16)[15::32,15::32]
    RGB = x_ @ srf

    HSI = torch.FloatTensor(HSI).T.unsqueeze(0).cuda()
    RGB = torch.FloatTensor(RGB).T.unsqueeze(0).cuda()
    R = f(HSI, RGB)
