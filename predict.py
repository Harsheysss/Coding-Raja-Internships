import torch
from model import SentimentModel
import pickle
import re
from transformers import AutoTokenizer

def preprocess_text(text, tokenizer):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return " ".join(tokenizer.tokenize(text))

def predict_sentiment(text, model, vectorizer, tokenizer):
    model.eval()
    preprocessed_text = preprocess_text(text, tokenizer)
    features = vectorizer.transform([preprocessed_text]).toarray()
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        sentiments = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiments[predicted.item()]

if __name__ == "__main__":
    input_dim = 10000  # Should match the input_dim used during training
    model = SentimentModel(input_dim)
    model.load_state_dict(torch.load('sentiment_model.pth'))

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    while True:
        text = input("Enter text to predict sentiment (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        sentiment = predict_sentiment(text, model, vectorizer, tokenizer)
        print(f"Sentiment: {sentiment}")
