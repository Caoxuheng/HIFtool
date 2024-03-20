import torch.nn as nn

class BandEmbed(nn.Module):
    """ Image to band Embedding
    """
    def __init__(self, hsi_channels, hsi_embed_dim):
        super().__init__()
        self.embed = nn.Conv2d(hsi_channels, hsi_embed_dim, 1, 1)
        self.norm = nn.LayerNorm(hsi_embed_dim)

    def forward(self, x):
        # x: [B, C, h, w]
        # return out: [B, D, h, w]
        embed_band = self.embed(x) # [B, D, h, w]
        out = self.norm(embed_band.permute(0, 2, 3, 1))  # [B, h, w, D]
        out = out.permute(0, 3, 1, 2)  # [B, D, h, w]

        return out
