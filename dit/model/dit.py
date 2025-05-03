

# diffusion transformer model.
# will
import torch
from model.encdec import Encoder, Decoder
from torch import nn
from dataclasses import dataclass
from einops import rearrange

@dataclass
class DitConfig:
    image_channels: int
    patch_size: int
    embed_dim: int
    num_patch_rows: int
    num_patch_cols: int
    num_classes: int
    # transformer params
    num_heads: int
    num_layers: int


class AdaLayerNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 3 * embed_dim),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.ln = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x, cond):
        h = self.ln(x)
        cond = self.mlp(cond)
        gamma, beta, alpha = cond.chunk(3, dim=-1)
        h = (1 + gamma) * h + beta

        return h, (1 + alpha)

# for this, we will use Llama style attention blocks.
class TransformerBlock(nn.Module):
    def __init__(self, 
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()

        self.attn_norm = AdaLayerNorm(embed_dim)
        self.mlp_norm = AdaLayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x, t):
        h, alpha = self.attn_norm(x, t)                                   # (B, S, D)
        h, _ = self.attn(h, h, h, average_attn_weights=False)   # (B, S, D) 
        h = h * alpha
        x = h + x 

        h, alpha = self.mlp_norm(x, t)
        x = self.mlp(h) * alpha + x                      # (B, S, D)
        return x

        
        

# a DiT - as simple as that.
class DiT(nn.Module):
    def __init__(self, 
        config: DitConfig,
    ):
        super().__init__()

        self.encoder = Encoder(
            patch_size=config.patch_size,
            image_channels=config.image_channels,
            embed_dim=config.embed_dim,
            num_patch_rows=config.num_patch_rows,
            num_patch_cols=config.num_patch_cols,
        )

        self.c_emb = nn.Embedding(config.num_classes, config.embed_dim)
        self.cond_emb = nn.Sequential(
            nn.Linear(1 + config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
        )

        self.decoder = Decoder(
            patch_size=config.patch_size,
            image_channels=config.image_channels,
            embed_dim=config.embed_dim,
            num_patch_rows=config.num_patch_rows,
            num_patch_cols=config.num_patch_cols,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
            )
            for _ in range(config.num_layers)
        ])

    def from_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
    
    def forward(self, x, t, c):
        # x: (B, C, H, W)
        # t: (B,)
        # c: (B,)
        x = self.encoder(x)
        
        # t: (B,) -> (B, 1) -> (B, D)) -> (B, 1, D)
        # cond_vec: (B, D)
        c_vec = self.c_emb(c) # (B, D)
        cond_vec = self.cond_emb(torch.cat([t.unsqueeze(-1), c_vec], dim=-1))
        # cond_vec: (B, D)
        cond = cond_vec.unsqueeze(1)

        for block in self.blocks:
            x = block(x, cond)

        x = self.decoder(x)
        return x
    
def test_dit():
    mnist_config = DitConfig(
        patch_size=2,
        num_layers=6,
        num_heads=4,
        embed_dim=64,
        image_channels=1,
        num_patch_rows=14,
        num_patch_cols=14
    )
    dit = DiT(mnist_config)

    x = torch.randn(1, 1, 28, 28)
    t = torch.tensor([0])
    y = dit(x, t)

if __name__ == "__main__":
    test_dit()