import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
import cv2
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class PSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_qy = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_ky = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescaley = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, y):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        b, hy, wy, cy = y.shape
        x = x_in.reshape(b, h * w, c)
        y = y.reshape(b, hy * wy, cy)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q_inpy = self.to_qy(y)
        k_inpy = self.to_ky(y)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        qy, ky = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inpy, k_inpy))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        qy = qy.transpose(-2, -1)
        ky = ky.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        qy = F.normalize(qy, dim=-1, p=2)
        ky = F.normalize(ky, dim=-1, p=2)

        attny = (ky @ qy.transpose(-2, -1))  # A = K^T*Q
        attny = attny * self.rescaley
        attny = attny.softmax(dim=-1)

        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attny @ attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class PSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, y):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, y) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class SpaD(nn.Module):
    def __init__(self, outband):
        super(SpaD, self).__init__()
        self.LRSD = nn.Sequential(nn.Conv2d(outband, 1, kernel_size=3, padding=1), nn.ReLU(),
                                  nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.conv_ = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.RT = nn.Sequential(nn.ReLU(), nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

    def forward(self, z, y):
        By = self.LRSD(y)
        By_ = F.interpolate(By, size=(z.shape[2], z.shape[3]), mode='bilinear', align_corners=False)
        B, C, H, W = z.shape
        out = self.conv_((z * By_).reshape(B * C, 1, H, W)).reshape(B, C, H, W)
        out = F.interpolate(out, size=(y.shape[2], y.shape[3]), mode='bilinear', align_corners=False)
        out = self.RT(out) * By + F.interpolate(z, size=(y.shape[2], y.shape[3]), mode='bilinear', align_corners=False)
        return out


class SpeD(nn.Module):
    def __init__(self, inband, outband):
        super(SpeD, self).__init__()

        self.dwconv = nn.Sequential(nn.Conv2d(inband, inband, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.wx_linear = nn.Sequential(nn.Linear(inband, inband), nn.Sigmoid())
        self.wx_expand = nn.Linear(inband, outband)

        self.z_conv1 = nn.Sequential(nn.Conv2d(outband, inband, kernel_size=1), nn.ReLU())
        self.z_depthwise = nn.Sequential(nn.Conv2d(inband, inband, kernel_size=3, stride=1, padding=1), nn.ReLU())

        self.final_conv = nn.Conv2d(outband, inband, kernel_size=1)

    def forward(self, z, x):
        B, C, H, W = x.size()
        _, c, _, _ = z.size()
        x_processed = self.dwconv(x)
        wx = self.wx_linear(x_processed.view(B, -1))  # [B,4,1,1] -> [B,4] -> [B,4]
        wx = wx.view(B, C, 1, 1)
        wx_expanded = self.wx_expand(wx.view(B, -1))  # [B,4] -> [B,128]
        wx_expanded = wx_expanded.view(B, c, 1, 1)  # [B,128,1,1]

        z_weighted = z * wx_expanded  # [B,128,H,W] * [B,128,1,1] -> [B,128,H,W]
        z_compressed = self.z_conv1(z_weighted)  # [B,4,H,W]
        z_processed = self.z_depthwise(z_compressed)  # [B,4,H,W]

        out = z_processed * wx  # [B,4,H,W] * [B,4,1,1] -> [B,4,H,W]
        out = self.final_conv(z) + out
        return out


class SpaU(nn.Module):
    def __init__(self, outband):
        super(SpaU, self).__init__()
        self.spa = nn.Sequential(nn.Conv2d(outband, outband, kernel_size=3, padding=1),
                                 nn.Conv2d(outband, outband, kernel_size=1), nn.ReLU())

    def forward(self, fin, x):
        out = self.spa(F.interpolate(fin, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False))
        return out


class SpeU(nn.Module):
    def __init__(self, inband, outband):
        super(SpeU, self).__init__()
        self.spe = nn.Sequential(nn.Conv2d(inband, outband, kernel_size=3, padding=1),
                                 nn.Conv2d(outband, outband, kernel_size=1), nn.ReLU())

    def forward(self, fin):
        out = self.spe(fin)
        return out


class DFTermPara(nn.Module):
    def __init__(self):
        super(DFTermPara, self).__init__()
        self.eta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, m, z):
        return 2 * self.eta * m + 2 * ((1 - self.eta) * z)


class DFTermErr(nn.Module):
    def __init__(self, inband, outband):
        super(DFTermErr, self).__init__()
        self.sped = SpeD(inband, outband)
        self.spad = SpaD(outband)

    def forward(self, z, x, y):
        spe_err = self.sped(z, x) - x
        spa_err = self.spad(z, y) - y
        return spe_err, spa_err


class DFTermFinal(nn.Module):
    def __init__(self, inband, outband):
        super(DFTermFinal, self).__init__()
        self.speu = SpeU(inband, outband)
        self.spau = SpaU(outband)

    def forward(self, m_out, spe_err, spa_err, x):
        out = m_out - self.speu(spe_err) - self.spau(spa_err, x)
        return out


class PST(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, dim=128, stage=2, num_blocks=[2, 4, 4]):
        super(PST, self).__init__()
        self.dim = dim
        self.stage = stage

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                PSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        self.bottleneck = PSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                PSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y_in):
        fea = self.embedding(x)
        fea_encoder = []
        y = y_in
        for (PSAB, FeaDownSample) in self.encoder_layers:
            fea = PSAB(fea, y)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            y = FeaDownSample(y)

        fea = self.bottleneck(fea, y)

        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            y = FeaUpSample(y)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = LeWinBlcok(fea, y)

        out = self.mapping(fea)
        return out


class PSTUN(nn.Module):
    def __init__(self, in_channels, in_feat, out_channels, stage=3):
        super(PSTUN, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, in_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.conv_y = nn.Conv2d(out_channels, in_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.body = nn.ModuleList([
            PST(in_dim=in_feat, out_dim=in_feat, dim=in_feat, stage=2, num_blocks=[1, 1, 1])
            for _ in range(stage)])

        self.para = nn.ModuleList([DFTermPara() for _ in range(stage)])
        self.ssr = DFTermErr(in_channels, in_feat)
        self.final = nn.ModuleList([DFTermFinal(in_channels, in_feat) for _ in range(stage)])

        self.conv_out = nn.Conv2d(in_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        PSTUN主网络
        Args:
            x: HRMSI输入 [B,c,H,W]
            y: LRHSI输入 [B,C,h,w]
        Returns:
            融合后HRHSI [B,C,H,W]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x_padded = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        z = self.conv_in(x_padded)
        y = self.conv_y(y)

        for i in range(self.stage):
            body = self.body[i]
            para = self.para[i]
            final = self.final[i]

            m = body(z, y)
            m_out = para(m, z)
            spe_err, spa_err = self.ssr(z, x, y)
            z = final(m_out, spe_err, spa_err, x)

        z = self.conv_out(z)
        return z[:, :, :h_inp, :w_inp]


if __name__ == "__main__":
    def my_summary(test_model, H=128, W=128, C=4, N=1):
        model = test_model.cuda()
        n_param = sum([p.nelement() for p in model.parameters()])
        print(f'Params:{n_param}')


    model = PSTUN().cuda()
    my_summary(model)
    # from thop import profile

    input_shape = (1, 4, 128, 128)  # 根据你的模型输入调整
    y_shape = (1, 128, 32, 32)  # 根据你的模型输入调整
    input_data = torch.randn(*input_shape).cuda()
    y = torch.randn(*y_shape).cuda()
    print(model(y, input_data).shape)
    # print(model(input_data,y).shape)
    # flops, params = profile(model, inputs=(y,input_data))
    # print(f"FLOPs: {flops/ (1024 * 1024 * 1024)}, Params: {params}")