# app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Model
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 2)
model.load_state_dict(torch.load("model/model.pth", map_location='cpu'))
model.eval()

st.title("🦸‍♂️ Superman vs Batman Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred = torch.argmax(outputs, dim=1).item()
        label = "Superman" if pred == 0 else "Batman"

    st.success(f"Prediction: {label}")
