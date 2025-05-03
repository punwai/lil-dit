import torch
from torch import nn
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_train_loader(
    batch_size=64, 
    dataset_size=None,
    distributed=False
):
    import torch.distributed as dist
    dataset = load_dataset("ylecun/mnist")
    shortened_dataset = dataset["train"]
    if dataset_size:
        shortened_dataset = shortened_dataset.select(range(dataset_size))

    def image_transform_2(x):
        x = transforms.ToTensor()(x)
        x = transforms.Resize((28, 28))(x)
        x = x.repeat(3, 1, 1)
        x = transforms.Normalize(0.5, 0.5)(x)
        return x.contiguous()

    shortened_dataset = shortened_dataset.map(
        lambda x: {
            "image_tensor": image_transform_2(x["image"]),
            "label_tensor": x["label"]
        },
        remove_columns=["image", "label"]
    ).with_format("torch")

    # Create a distributed sampler
    kwargs = {}
    if distributed:
        kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(
            shortened_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )

    train_loader = DataLoader(
        shortened_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        **kwargs
    )
    return train_loader

if __name__ == "__main__":
    train_loader = get_train_loader()
    for batch in train_loader:
        img_tensor = batch["image_tensor"]
        print("Min:", img_tensor.min().item(), "Max:", img_tensor.max().item(), "Mean:", img_tensor.mean().item(), "Std:", img_tensor.std().item())
        break
