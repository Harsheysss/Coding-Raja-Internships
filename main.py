import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from model import SentimentModel


class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


def load_and_preprocess_data(train_file, val_file):
    train_df = pd.read_csv(train_file, header=None, delimiter=',', usecols=[2, 3], names=['label', 'text'])
    val_df = pd.read_csv(val_file, header=None, delimiter=',', usecols=[2, 3], names=['label', 'text'])

    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    print("Train DataFrame before filtering:")
    print(train_df.head())
    print("\nValidation DataFrame before filtering:")
    print(val_df.head())

    # Filter out invalid labels
    valid_labels = {0, 1, 2}
    train_df = train_df[train_df['label'].isin(valid_labels)]
    val_df = val_df[val_df['label'].isin(valid_labels)]

    # Debug: Print the first few rows of the dataframes after filtering
    print("\nTrain DataFrame after filtering:")
    print(train_df.head())
    print("\nValidation DataFrame after filtering:")
    print(val_df.head())

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()

    return train_texts, train_labels, val_texts, val_labels


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for texts, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs = torch.tensor(texts, dtype=torch.float32).clone().detach()
            targets = torch.tensor(labels, dtype=torch.long).clone().detach()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


# Function to evaluate the model
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            inputs = torch.tensor(texts, dtype=torch.float32).clone().detach()
            targets = torch.tensor(labels, dtype=torch.long).clone().detach()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")


# Main function
if __name__ == "__main__":
    train_file = 'C:/Users/khatr/OneDrive/Desktop/Study/Internship Tings/Datasets/Sentiment Dataset 3/twitter_training.csv'
    val_file = 'C:/Users/khatr/OneDrive/Desktop/Study/Internship Tings/Datasets/Sentiment Dataset 3/twitter_validation.csv'

    train_texts, train_labels, val_texts, val_labels = load_and_preprocess_data(train_file, val_file)

    print(f"Number of training samples: {len(train_texts)}")
    print(f"Number of validation samples: {len(val_texts)}")

    vectorizer = TfidfVectorizer(max_features=10000)
    train_features = vectorizer.fit_transform(train_texts).toarray()
    val_features = vectorizer.transform(val_texts).toarray()

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)

    # Check unique labels
    print("Unique train labels:", set(train_labels))
    print("Unique validation labels:", set(val_labels))

    train_dataset = SentimentDataset(train_features, train_labels)
    val_dataset = SentimentDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim = train_features.shape[1]
    model = SentimentModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=5)
    evaluate_model(model, val_loader, criterion)

    # Save the trained model
    torch.save(model.state_dict(), 'sentiment_model.pth')
    torch.save(vectorizer, 'vectorizer.pth')
    torch.save(label_encoder, 'label_encoder.pth')