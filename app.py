import streamlit as st
from google.cloud import vision
from PIL import Image
import io
import json
import os

# Step 1: Set up Google Vision API credentials
def load_vision_client():
    # Load credentials from environment variable (set in Streamlit Cloud secrets)
    credentials_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    client = vision.ImageAnnotatorClient.from_service_account_info(credentials_info)
    return client

# Step 2: Function to analyze the uploaded image
def analyze_image(image_bytes):
    client = load_vision_client()
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    return [(label.description, label.score) for label in labels]

# Step 3: Streamlit App Layout
st.title("Image Analysis Web App with Google Vision API")
st.write("Upload an image to analyze it using Google Vision API")

# Step 4: Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Analyze the image using Google Vision API
    image_bytes = uploaded_file.read()
    labels = analyze_image(image_bytes)

    # Display the analysis results
    st.write("Detected Labels:")
    for label, score in labels:
        st.write(f"- {label}: {score:.2f}")
