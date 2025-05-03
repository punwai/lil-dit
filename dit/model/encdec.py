# a diffusion transformer model written from scratch.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import einops

# you should encode more complex frames.
class Encoder(nn.Module):
    def __init__(
        self, 
        patch_size: int,
        image_channels: int,
        embed_dim: int,
        num_patch_rows: int,
        num_patch_cols: int,
    ):
        super().__init__()
        # our patches should be 16x16
        self.patch_size = patch_size
        self.encoder = nn.Conv2d(
            image_channels, 
            embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size,
        )
        nn.init.trunc_normal_(self.encoder.weight, std=0.02)
        self.num_patch_rows = num_patch_rows
        self.num_patch_cols = num_patch_cols

        # self.E_row = nn.Parameter(torch.zeros(self.num_patch_rows, embed_dim))
        # self.E_col = nn.Parameter(torch.zeros(self.num_patch_cols, embed_dim))
        self.pwe = nn.Parameter(torch.zeros(self.num_patch_rows * self.num_patch_cols, embed_dim))
    
    # INPUT
    # x: [B, C, H, W]
    def forward(self, x):
        # b d h w -> b h w d
        h = self.encoder(x).permute(0, 2, 3, 1)
        h = einops.rearrange(h, "b h w d -> b (h w) d") + self.pwe[None, :, :]
        return h

# decode the tokens
class Decoder(nn.Module):
    def __init__(self, 
        patch_size: int,
        image_channels: int,
        embed_dim: int,
        num_patch_rows: int,
        num_patch_cols: int,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.decoder = nn.ConvTranspose2d(
            embed_dim, 
            image_channels, 
            kernel_size=self.patch_size, 
            stride=self.patch_size,
        )
        nn.init.trunc_normal_(self.decoder.weight, std=0.02)
        self.num_patch_rows = num_patch_rows
        self.num_patch_cols = num_patch_cols

    # INPUT
    # x: [B, embed_dim, H_p, W_p]
    def forward(self, x):
        # b (h w) d -> b d h w
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=self.num_patch_rows, w=self.num_patch_cols)
        # b d h w -> b c h w
        return self.decoder(x)

# small test
if __name__ == "__main__":
    random_image = torch.randn(1, 3, 64, 64)
    encoder = Encoder(
        patch_size=4, 
        image_channels=3, 
        embed_dim=128, 
        num_patch_rows=16, 
        num_patch_cols=16
    )
    h = encoder(random_image)
