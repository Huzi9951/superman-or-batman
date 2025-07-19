import torch
from model import VGG
from torchvision import transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model with correct parameters
model = VGG(input_shape=3, hidden_units=30, output_layer=2).to(device)

# Load state dict
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.eval()

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Inference example
def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# Example usage
if __name__ == "__main__":
    prediction = predict("test_img.jpg")
    print("Predicted class:", prediction)

