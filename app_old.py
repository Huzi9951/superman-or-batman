import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom VGG-like model definition (must match training)
class VGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_layer):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, output_layer)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
model = VGG(input_shape=3, hidden_units=30, output_layer=2).to(device)
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.eval()

# Transformations for input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Function to predict class of image
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_names = ["batman", "superman"]
    return class_names[predicted.item()]

# For testing
if __name__ == "__main__":
    test_image = "test/batman.jpg"  # Replace with your test image path
    prediction = predict(test_image)
    print(f"Prediction: {prediction}")


