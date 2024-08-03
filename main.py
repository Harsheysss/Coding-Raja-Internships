import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
from tqdm import tqdm
import os
from data_loader import get_data_loaders
from model import get_model


def train_and_validate(num_epochs=10, batch_size=8, learning_rate=0.001,
                       data_dir='C:/Users/khatr/OneDrive/Desktop/Study/Internship Tings/Food102'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_data_loaders(batch_size, data_dir)

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    model_save_path = 'saved_models'
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        end_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s')

        validate(model, val_loader, criterion, device)

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch + 1}.pth'))


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    train_and_validate()