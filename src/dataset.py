# src/dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import IMAGE_SIZE, BATCH_SIZE

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

def get_dataloader(data_dir, train=True):
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms(train)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train
    )

    return loader, dataset.classes
