import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import VGG

# Set up the page
st.set_page_config(
    page_title="Superman vs Batman Classifier",
    page_icon="🦸‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #FFD700;
            padding-bottom: 10px;
        }
        .hero-header {
            background: linear-gradient(to right, #1E90FF, #2F4F4F);
            padding: 20px;
            border-radius: 12px;
        }
        .upload-box {
            background-color: #00000022;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }
        .prediction {
            font-size: 30px;
            text-align: center;
            margin-top: 20px;
            background-color: #FFD70011;
            padding: 15px;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="hero-header"><div class="title">🦸‍♂️ Superman vs Batman Classifier</div></div>', unsafe_allow_html=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG(input_shape=3, hidden_units=30, output_layer=2)  # 2 classes: Superman & Batman
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.to(device)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Upload image
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image of Superman or Batman", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform and predict
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred_class = output.max(1)

    classes = ["Batman 🦇", "Superman 🦸‍♂️"]
    prediction = classes[pred_class.item()]

    # Show prediction
    st.markdown(f'<div class="prediction">Prediction: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
