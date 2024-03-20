import torch
import torch.nn as nn
import torch.nn.functional as F
from .HSI.SpeT import SpeTL
from .MSI.SpaT import SpaTL
from timm.models.layers import trunc_normal_
from .HSI.band_embed import BandEmbed
from .MSI.patch_embed import PatchEmbed

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x1)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x
        return out

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        # shallow feature extraction
        self.shallow1 = nn.Conv2d(args.hsi_channel + args.msi_channel, args.n_feats, 3, 1, 1)
        self.shallow2 = nn.Conv2d(args.hsi_channel + args.msi_channel, args.n_feats, 3, 1, 1)

        self.RB1 = nn.ModuleList()
        self.RB2 = nn.ModuleList()
        for i in range(self.args.hsi_res_blocks):
            self.RB1.append(ResBlock(self.args.n_feats, self.args.n_feats))
        for i in range(self.args.msi_res_blocks):
            self.RB2.append(ResBlock(self.args.n_feats, self.args.n_feats))
        self.conv_tail1 = conv3x3(self.args.n_feats, self.args.hsi_channel)
        self.conv_tail2 = conv3x3(self.args.n_feats, self.args.hsi_channel)

        # band_embedding_1
        self.band_embedding_1 = BandEmbed(args.hsi_channel, args.hsi_embed_dim)

        self.spectral_layers_1 = SpeTL(dim=args.hsi_embed_dim, heads=args.hsi_heads, num_layers=args.hsi_num_layers)
        self.norm1 = nn.LayerNorm(args.hsi_embed_dim)
        self.band_unembed_1 = nn.Conv2d(args.hsi_embed_dim, args.hsi_channel, 1, 1)
        self.conv1 = nn.Conv2d(args.hsi_channel, args.hsi_channel, 3, 1, 1)

        # band_embedding_2
        self.band_embedding_2 = BandEmbed(args.hsi_channel, args.hsi_embed_dim // 2)

        self.spectral_layers_2 = SpeTL(dim=args.hsi_embed_dim // 2, heads=args.hsi_heads, num_layers=args.hsi_num_layers)
        self.norm2 = nn.LayerNorm(args.hsi_embed_dim // 2)
        self.band_unembed_2 = nn.Conv2d(args.hsi_embed_dim // 2, args.hsi_channel, 1, 1)
        self.conv2 = nn.Conv2d(args.hsi_channel, args.hsi_channel, 3, 1, 1)

        # band_embedding_3
        self.band_embedding_3 = BandEmbed(args.hsi_channel, args.hsi_embed_dim * 2)

        self.spectral_layers_3 = SpeTL(dim=args.hsi_embed_dim * 2, heads=args.hsi_heads,num_layers=args.hsi_num_layers)
        self.norm3 = nn.LayerNorm(args.hsi_embed_dim * 2)
        self.band_unembed_3 = nn.Conv2d(args.hsi_embed_dim * 2, args.hsi_channel, 1, 1)
        self.conv3 = nn.Conv2d(args.hsi_channel, args.hsi_channel, 3, 1, 1)


        # patch_embedding_16
        self.patch_embedding_16 = PatchEmbed(args.hsi_channel, args.msi_embed_dim, args.patch_size_n)
        self.spatial_layers_16 = SpaTL(embed_dim=args.msi_embed_dim, heads=args.msi_heads, num_layers=args.msi_num_layers)
        self.norm_16 = nn.LayerNorm(args.msi_embed_dim)
        # patch_unembed
        self.patch_unembed_16 = nn.ConvTranspose2d(args.msi_embed_dim, args.hsi_channel, kernel_size=args.patch_size_n, stride=args.patch_size_n)
        self.conv_16 = nn.Conv2d(args.hsi_channel, args.hsi_channel, 3, 1, 1, bias=False)

        # patch_embedding_8
        self.patch_embedding_8 = PatchEmbed(args.hsi_channel, args.msi_embed_dim, args.patch_size_n // 2)
        self.spatial_layers_8 = SpaTL(embed_dim=args.msi_embed_dim, heads=args.msi_heads, num_layers=args.msi_num_layers)
        self.norm_8 = nn.LayerNorm(args.msi_embed_dim)
        # patch_unembed
        self.patch_unembed_8 = nn.ConvTranspose2d(args.msi_embed_dim, args.hsi_channel, kernel_size=args.patch_size_n // 2, stride=args.patch_size_n // 2)
        self.conv_8 = nn.Conv2d(args.hsi_channel, args.hsi_channel, 3, 1, 1, bias=False)

        # patch_embedding_32
        self.patch_embedding_32 = PatchEmbed(args.hsi_channel, args.msi_embed_dim, args.patch_size_n * 2)
        self.spatial_layers_32 = SpaTL(embed_dim=args.msi_embed_dim, heads=args.msi_heads, num_layers=args.msi_num_layers)
        self.norm_32 = nn.LayerNorm(args.msi_embed_dim)
        # patch_unembed
        self.patch_unembed_32 = nn.ConvTranspose2d(args.msi_embed_dim, args.hsi_channel, kernel_size=args.patch_size_n * 2, stride=args.patch_size_n * 2)
        self.conv_32 = nn.Conv2d(args.hsi_channel, args.hsi_channel, 3, 1, 1, bias=False)

        #upsample
        self.upsample = nn.Sequential(
            nn.Conv2d(args.hsi_channel, args.n_feats, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(args.n_feats, args.sf*args.sf*args.n_feats, 3, 1, 1),
            nn.PixelShuffle(args.sf),
            nn.Conv2d(args.n_feats, args.hsi_channel, 3, 1, 1)
        )
        #reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(args.hsi_channel*2, args.n_feats, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(args.n_feats, args.hsi_channel, 3, 1, 1)
        )

        self.a = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).to(args.device)
        self.b = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).to(args.device)
        self.c = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).to(args.device)

        self.a1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).to(args.device)
        self.b1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).to(args.device)
        self.c1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).to(args.device)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, y, z):
        #y: [B, C, h, w]   z :[B, c, H, W]
        #x_5: [B, C, H, W]
        y_up = F.interpolate(y, scale_factor=self.args.sf, mode='bilinear')  #[B, C, H, W]
        z_down = F.interpolate(z, scale_factor=1 / self.args.sf, mode='bilinear')  #[B, c, h, w]

        z_1 = self.shallow1(torch.cat((y, z_down), dim=1)) #[B, C, h, w]
        x_1 = self.shallow2(torch.cat((y_up, z), dim=1))  # [B, C, H, W]

        z1_res = z_1
        for i in range(self.args.hsi_res_blocks):
            z1_res = self.RB1[i](z1_res)
        z_1 = z_1 + z1_res
        z_1 = self.conv_tail1(z_1)

        x1_res = x_1
        for i in range(self.args.msi_res_blocks):
            x1_res = self.RB2[i](x1_res)
        x_1 = x_1 + x1_res
        x_1 = self.conv_tail2(x_1)

        embed_band_1 = self.band_embedding_1(z_1) #[B, C, h, w]
        _, C_1, h, w = embed_band_1.shape
        # position embedding
        pos_embed_hsi_1 = nn.Parameter(torch.zeros(1, C_1, h, w)).to(self.args.device)
        trunc_normal_(pos_embed_hsi_1, std=.02)
        embed_band_1 = embed_band_1 + pos_embed_hsi_1  # [B, C, h, w]
        
        fea_hsi_1 = self.spectral_layers_1(embed_band_1)  # [B, C, h, w]
        fea_hsi_1 = self.norm1(fea_hsi_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #[B, C, h, w]
        x_hsi_1 = self.band_unembed_1(fea_hsi_1) #[B, C, h, w]
        x_hsi_1 = self.conv1(x_hsi_1)  # [B, C, h, w]


        embed_band_2 = self.band_embedding_2(z_1)  # [B, C/2, h, w]
        _, C_2, _, _ = embed_band_2.shape
        # position embedding
        pos_embed_hsi_2 = nn.Parameter(torch.zeros(1, C_2, h, w)).to(self.args.device)
        trunc_normal_(pos_embed_hsi_2, std=.02)
        embed_band_2 = embed_band_2 + pos_embed_hsi_2  # [B, C/2, h, w]
        
        fea_hsi_2 = self.spectral_layers_2(embed_band_2)  # [B, C/2, h, w]
        fea_hsi_2 = self.norm2(fea_hsi_2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, C/2, h, w]
        x_hsi_2 = self.band_unembed_2(fea_hsi_2)  # [B, C/2, h, w]
        x_hsi_2 = self.conv2(x_hsi_2)  # [B, C, h, w]


        embed_band_3 = self.band_embedding_3(z_1)  # [B, C*2, h, w]
        _, C_3, _, _ = embed_band_3.shape
        # position embedding
        pos_embed_hsi_3 = nn.Parameter(torch.zeros(1, C_3, h, w)).to(self.args.device)
        trunc_normal_(pos_embed_hsi_3, std=.02)
        embed_band_3 = embed_band_3 + pos_embed_hsi_3  # [B, C*2, h, w]
        
        fea_hsi_3 = self.spectral_layers_3(embed_band_3)  # [B, C*2, h, w]
        fea_hsi_3 = self.norm3(fea_hsi_3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, C*2, h, w]
        x_hsi_3 = self.band_unembed_3(fea_hsi_3)  # [B, C*2, h, w]
        x_hsi_3 = self.conv3(x_hsi_3)  # [B, C, h, w]

        sum = self.a + self.b + self.c
        x_hsi = x_hsi_1 * (self.a / sum) + x_hsi_2 * (self.b / sum) + x_hsi_3 * (self.c / sum)  # [B, C, h, w]
        z_2 = z_1 + x_hsi    # [B, C, h, w]

        x_2 = self.upsample(z_2)   # [B, C, H, W]


        embed_patch_16 = self.patch_embedding_16(x_1)   # [B, d, H, W]
        _, d, n_h_16, n_w_16 = embed_patch_16.shape
        # position embedding
        pos_embed_msi_16 = nn.Parameter(torch.zeros(1, d, n_h_16, n_w_16)).to(self.args.device)
        trunc_normal_(pos_embed_msi_16, std=.02)
        embed_patch_16 = embed_patch_16 + pos_embed_msi_16  # [B, d, H, W]
        
        fea_msi_16 = self.spatial_layers_16(embed_patch_16) #[B, d, H, W]
        fea_msi_16 = self.norm_16(fea_msi_16.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #[B, d, H, W]

        x_msi_16 = self.patch_unembed_16(fea_msi_16) #[B, C, H, W]

        x_msi_16 = self.conv2(x_msi_16) #[B, C, H, W]

        embed_patch_8 = self.patch_embedding_8(x_1)  # [B, d, H*2, W*2]
        _, _, n_h_8, n_w_8 = embed_patch_8.shape
        # position embedding
        pos_embed_msi_8 = nn.Parameter(torch.zeros(1, d, n_h_8, n_w_8)).to(self.args.device)
        trunc_normal_(pos_embed_msi_8, std=.02)
        embed_patch_8 = embed_patch_8 + pos_embed_msi_8  # [B, d, H*2, W*2]
        
        fea_msi_8 = self.spatial_layers_8(embed_patch_8)  # [B, d, H*2, W*2]
        fea_msi_8 = self.norm_8(fea_msi_8.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, d, H/patch, W/patch]
        x_msi_8 = self.patch_unembed_8(fea_msi_8)  # [B, C, H, W]
        x_msi_8 = self.conv_8(x_msi_8)  # [B, C, H, W]

        embed_patch_32 = self.patch_embedding_32(x_1)  # [B, d, H/2, W/2]
        _, _, n_h_32, n_w_32 = embed_patch_32.shape
        # position embedding
        pos_embed_msi_32 = nn.Parameter(torch.zeros(1, d, n_h_32, n_w_32)).to(self.args.device)
        trunc_normal_(pos_embed_msi_32, std=.02)
        embed_patch_32 = embed_patch_32 + pos_embed_msi_32  # [B, d, H/2, W/2]
        
        fea_msi_32 = self.spatial_layers_32(embed_patch_32)  # [B, d, H/2, W/2]
        fea_msi_32 = self.norm_32(fea_msi_32.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, d, H/2, W/2]
        x_msi_32 = self.patch_unembed_32(fea_msi_32)  # [B, C, H, W]
        x_msi_32 = self.conv_32(x_msi_32)  # [B, C, H, W]

        sum1 = self.a1 + self.b1 + self.c1

        x_msi = x_msi_16 * (self.a1 / sum1) + x_msi_8 * (self.b1 / sum1) + x_msi_32 * (self.c1 / sum1)   # [B, C, H, W]

        x_3 = x_1 + x_msi   # [B, C, H, W]

        x_4 = torch.cat((x_2, x_3), dim=1)   # [B, C*2, H, W]

        x_5 = self.recon(x_4)   # [B, C, H, W]

        return x_5
