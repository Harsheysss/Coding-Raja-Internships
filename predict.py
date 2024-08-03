import torch
from torchvision import transforms
from PIL import Image
import json
from model import get_model

# Load the class names (adjust the path to your actual file)
with open('C:/Users/khatr/PycharmProjects/Food 5 Final/classes.json', 'r') as f:
    class_names = json.load(f)

def load_model(model_path):
    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Change to (1024, 1024) if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[str(predicted.item())]

if __name__ == "__main__":
    # Load the model
    model_path = "C:/Users/khatr/PycharmProjects/Food 5 Final/saved_models/model_epoch_10.pth"  # Replace with the path to your saved model
    model = load_model(model_path)

    # Preprocess the image
    image_path = "C:/Users/khatr/Downloads/Images/Food/Garlic Bread.jpeg"  # Replace with the path to the image you want to predict
    image_tensor = preprocess_image(image_path)

    # Perform prediction
    class_label = predict_image(model, image_tensor, class_names)
    print(f'The image is predicted to be: {class_label}')
