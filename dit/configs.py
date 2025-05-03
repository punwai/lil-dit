from dataclasses import dataclass
from model.dit import DitConfig

def compute_params(config: DitConfig):
    # mlp per layer
    total_mlps = config.num_layers * (config.embed_dim**2 * 4)
    total_attn = config.num_layers * (config.embed_dim**2) * 4
    return total_mlps + total_attn

mnist_config = DitConfig(
    image_channels=3,
    patch_size=2,
    embed_dim=128,
    num_patch_rows=14,
    num_patch_cols=14,
    num_heads=4,
    num_layers=8,
    num_classes=10,
)

# mnist_config = DitConfig(
#     image_channels=3,
#     patch_size=4,
#     embed_dim=128,
#     num_patch_rows=14,
#     num_patch_cols=14,
#     num_heads=8,
#     num_layers=12,
# )

if __name__ == "__main__":
    print(compute_params(mnist_config))
