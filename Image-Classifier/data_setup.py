
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

NUM_WORKERS = 0  # Suitable for Windows

def create_dataloaders(
    train_dir,
    test_dir,
    batch_size=16
):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataloader, test_dataloader, train_dataset.classes
