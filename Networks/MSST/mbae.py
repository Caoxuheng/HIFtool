import torch
import torch.nn as nn
from Networks.MSST.HSI.SpeT import SpeTL
from Networks.MSST.HSI.mask_band import MaskEmbed
from timm.models.layers import trunc_normal_
from Networks.MSST.HSI.band_embed import BandEmbed

class MBAEViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed = BandEmbed(args.hsi_channel, args.hsi_embed_dim)
        self.encoder = SpeTL(dim=args.hsi_embed_dim, heads=args.hsi_heads, num_layers=args.hsi_num_layers)
        self.norm1 = nn.LayerNorm(args.hsi_embed_dim)
        self.proj = nn.Linear(args.hsi_embed_dim, args.hsi_embed_dim//2)
        self.decoder = SpeTL(dim=args.hsi_embed_dim//2, heads=args.hsi_heads, num_layers=2)

        self.restruction = nn.Linear(args.hsi_embed_dim//2, args.hsi_embed_dim)
        self.norm2 = nn.LayerNorm(args.hsi_embed_dim)
        # restore image from unconv
        self.unembed = nn.Conv2d(args.hsi_embed_dim, args.hsi_channel, 1, 1)
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

    def forward(self, y):
        #y: [B, C, h, w]
        #restore_image: [B, C, h, w]
        embed_band = self.embed(y) #[B, D, h, w]
        B, D, h, w = embed_band.shape
        # position embedding
        pos_embed_hsi = nn.Parameter(torch.zeros(1, D, h, w)).to(self.args.hsi_device)
        trunc_normal_(pos_embed_hsi, std=.02)
        embed_band_norm = embed_band + pos_embed_hsi  # [B, D, h, w]

        encoder_embed, sample_index, mask_index = MaskEmbed(embed_band_norm, self.args.hsi_mask_ratio)
        fea_hsi = self.encoder(encoder_embed)  # [B, D, h, w]
        fea_hsi_norm = self.norm1(fea_hsi.permute(0, 2, 3, 1)) # [B, h, w, C]
        fea_hsi_proj = self.proj(fea_hsi_norm.reshape(-1, h*w, D)).view(B, h, w, -1) #[B, h, w, D/2]
        fea_hsi_proj = fea_hsi_proj.permute(0, 3, 1, 2) # [B, D/2, h, w]

        _, d, _, _ = fea_hsi_proj.shape
        # position embedding
        pos_mask_embed_hsi = nn.Parameter(torch.zeros(1, d, h, w)).to(self.args.hsi_device)
        trunc_normal_(pos_mask_embed_hsi, std=.02)
        decoder_embed = fea_hsi_proj + pos_mask_embed_hsi  # [B, D/2, h, w]
        
        fea_hsi_de = self.decoder(decoder_embed)  # [B, D/2, h, w]
        outputs = self.restruction(fea_hsi_de.permute(0, 2, 3, 1).reshape(-1, h*w, d)) #[B, h*w, D]
        output_norm = self.norm2(outputs.view(-1, h, w, D)).permute(0, 3, 1, 2) #[B, D, h, w]

        restore_image = self.unembed(output_norm) #[B, C, h, w]

        return restore_image

