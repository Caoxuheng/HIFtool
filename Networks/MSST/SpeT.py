import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        # self.to_q = nn.Linear(dim, dim)
        # self.to_k = nn.Linear(dim, dim)
        # self.to_v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)
        # self.pos_emb = nn.Sequential(
        #     nn.Conv2d(dim_head * heads, dim, 3, 1, 1),
        #     GELU(),
        #     nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
        # )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)

        #qkv = self.qkv(x).reshape(b, h * w // self.num_heads, 3, self.num_heads, c).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(b, h * w // self.num_heads, self.num_heads, 3, c).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, heads, h*w/heads, c]
        # q = self.to_q(x).reshape(b, h * w // self.num_heads, self.num_heads, c).permute(0, 2, 1, 3) #[b, heads, h*w/heads, c]
        # k = self.to_k(x).reshape(b, h * w // self.num_heads, self.num_heads, c).permute(0, 2, 1, 3)
        # v = self.to_v(x).reshape(b, h * w // self.num_heads, self.num_heads, c).permute(0, 2, 1, 3)
        q = q.transpose(-2, -1) # [b, heads, c, h*w/heads]
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        # q = F.normalize(q, dim=-1, p=2)
        # k = F.normalize(k, dim=-1, p=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale # [b, heads, c, c]
        attn = attn.softmax(dim=-1)
        x = attn @ v   # [b, heads, c, hw/heads]
        x = x.permute(0, 3, 1, 2)    # [b, hw/heads, heads, c]
        x = x.reshape(b, h * w, c)
        out_c = self.proj(x).view(b, h, w, c)
        #out_p = self.pos_emb(v.permute(0, 3, 1, 2).reshape(b,c,h,w)).permute(0, 2, 3, 1)
        out = out_c

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
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.mlp(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class SpeTL(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            num_layers,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads=heads),
                nn.LayerNorm(dim),
                ConvMlp(dim)
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (norm1, attn, norm2, mlp) in self.blocks:
            x = attn(norm1(x)) + x
            x = mlp(norm2(x)) + x
        out = x.permute(0, 3, 1, 2)
        return out
