import torch.nn as nn

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, msi_channels, msi_embed_dim, patch_size):
        super().__init__()
        self.embed = nn.Conv2d(msi_channels, msi_embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(msi_embed_dim)

    def forward(self, x):
        # x: [B, c, H, W]
        # return out: [B, D, h, w]
        out = self.embed(x)  # [B, D, h, w]
        out = self.norm(out.permute(0, 2, 3, 1))  # [B, h, w, D]
        out = out.permute(0, 3, 1, 2)  # [B, D, h, w]
        return out
