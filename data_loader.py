import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size, data_dir='C:/Users/khatr/OneDrive/Desktop/Study/Internship Tings/Food102'):
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Change to (1024, 1024) if needed
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Change to (1024, 1024) if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=f'{data_dir}/images/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/images/test', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader