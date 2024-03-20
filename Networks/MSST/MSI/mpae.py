import torch
import torch.nn as nn
from Networks.MSST.MSI.SpaT import SpaTL
from Networks.MSST.MSI.mask_patch import MaskEmbed
from timm.models.layers import trunc_normal_
from Networks.MSST.MSI.patch_embed import PatchEmbed


class MPAEViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.embed = nn.Conv2d(args.msi_channel, args.msi_embed_dim, kernel_size=args.patch_size, stride=args.patch_size)
        # self.norm1 = nn.LayerNorm(args.msi_embed_dim)
        self.embed = PatchEmbed(args.msi_channel, args.msi_embed_dim, args.patch_size)
        self.encoder = SpaTL(embed_dim=args.msi_embed_dim, heads=args.msi_heads, num_layers=args.msi_num_layers)
        self.norm1 = nn.LayerNorm(args.msi_embed_dim)
        self.proj = nn.Linear(args.msi_embed_dim, args.msi_embed_dim // 2)
        self.decoder = SpaTL(embed_dim=args.msi_embed_dim // 2, heads=args.msi_heads, num_layers=2)

        self.restruction = nn.Linear(args.msi_embed_dim // 2, args.msi_embed_dim)
        self.norm2 = nn.LayerNorm(args.msi_embed_dim)
        # restore image from unconv
        self.unembed = nn.ConvTranspose2d(args.msi_embed_dim, args.msi_channel, kernel_size=args.patch_size, stride=args.patch_size)
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
    def forward(self, z):
        #z: [B, c, H, W]
        #restore_image : [B, c, h, w]
        embed_patch = self.embed(z) # [B, D, h, w]
        B, D, h, w = embed_patch.shape
        # position embedding
        pos_embed_msi = nn.Parameter(torch.zeros(1, D, h, w)).to(self.args.msi_device)
        trunc_normal_(pos_embed_msi, std=.02)
        embed_patch_norm = embed_patch + pos_embed_msi  # [B, D, h, w]
        encoder_embed, sample_index, mask_index = MaskEmbed(embed_patch_norm, self.args.msi_mask_ratio)
        fea_msi = self.encoder(encoder_embed)  # [B, D, h, w]
        fea_msi_norm = self.norm1(fea_msi.permute(0, 2, 3, 1))  # [B, h, w, D]
        fea_msi_proj = self.proj(fea_msi_norm.reshape(-1, h * w, D)).view(B, h, w, -1)  # [B, h, w, D/2]
        fea_msi_proj = fea_msi_proj.permute(0, 3, 1, 2)  # [B, D/2, h, w]

        _, d, h, w = fea_msi_proj.shape
        # position embedding
        pos_mask_embed_msi = nn.Parameter(torch.zeros(1, d, h, w)).to(self.args.msi_device)
        trunc_normal_(pos_mask_embed_msi, std=.02)
        decoder_embed = fea_msi_proj + pos_mask_embed_msi  # [B, D/2, h, w]
        
        fea_msi_de = self.decoder(decoder_embed)  # [B, D/2, h, w]
        outputs = self.restruction(fea_msi_de.permute(0, 2, 3, 1).reshape(-1, h * w, d))  # [B, h*w, D]
        output_norm = self.norm2(outputs.view(-1, h, w, D)).permute(0, 3, 1, 2)  # [B, D, h, w]

        restore_image = self.unembed(output_norm) # [B, c, h, w]

        return restore_image

