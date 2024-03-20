import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Attention(nn.Module):
    def __init__(
            self,
            embed_dim,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = embed_dim // heads
        self.embed_dim = embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.scale = head_dim ** -0.5
        self.proj = nn.Linear(embed_dim, embed_dim)
        # self.pos_emb = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),
        #     GELU(),
        #     nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),
        # )

    def forward(self, x_in):
        """
        x_in: [b,h,w,d]
        return out: [b,h,w,d]
        """
        b, h, w, d = x_in.shape
        x = x_in.reshape(b, h * w, d)  # b,hw,d -> b,n,d

        qkv = self.qkv(x).reshape(b, h*w, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  #[b, heads, h*w, d/heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale #[b, heads, h*w, h*w]
        attn = attn.softmax(dim=-1)
        x = (attn @ v) #[b, heads, h*w, d/heads]
        x = x.permute(0, 2, 1, 3) #[b, h*w, d/heads, heads]
        x = x.reshape(b, h*w, d)
        out_s = self.proj(x).reshape(b, h, w, d)
        #out_p = self.pos_emb(v.permute(0, 3, 1, 2).reshape(b, self.embed_dim, h, w)).permute(0, 2, 3, 1)
        out = out_s

        return out

class ConvMlp(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1)
        )

    def forward(self, x):
        """
        x: [b,h,w,d]
        return out: [b,h,w,d]
        """
        out = self.mlp(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class SpaTL(nn.Module):
    def __init__(
            self,
            embed_dim,
            heads,
            num_layers,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(embed_dim),
                Attention(embed_dim, heads=heads),
                nn.LayerNorm(embed_dim),
                ConvMlp(embed_dim)
            ]))

    def forward(self, x):
        """
        x: [b,d,h,w]
        return out: [b,d,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (norm1, attn, norm2, mlp) in self.blocks:
            x = attn(norm1(x)) + x
            x = mlp(norm2(x)) + x
        out = x.permute(0, 3, 1, 2)
        return out