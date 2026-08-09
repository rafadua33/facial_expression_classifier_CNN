"""
app.py

Streamlit app for the FER2013 facial expression classifier.
Lets the user upload a photo (or use their webcam) and shows the
model's predicted emotion along with confidence scores for all classes.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
import torch
from PIL import Image

# src/ isn't a package with an __init__.py, so we add it to sys.path
# directly rather than doing "from src.inference import ...". This lets
# inference.py's own internal "from model import EmotionCNN" resolve too.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference import load_model, predict_emotion  # noqa: E402

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "best.pt")


@st.cache_resource
def get_model():
    """
    Loads the model once and caches it across reruns/interactions.
    Without this, Streamlit would reload the model from disk on every
    single button click or file upload, which is slow and unnecessary
    since the weights never change during a session.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device)
    return model, device


st.set_page_config(page_title="Facial Expression Classifier", page_icon=":)")
st.title("Facial Expression Classifier")
st.write(
    "Upload a photo or take one with your camera. The model will detect "
    "the face and predict the emotion being expressed."
)

model, device = get_model()

tab_upload, tab_camera = st.tabs(["Upload a photo", "Use camera"])

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = uploaded_file

with tab_camera:
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        image_source = camera_photo

if image_source is not None:
    image = Image.open(image_source)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input photo", use_container_width=True)

    with st.spinner("Classifying..."):
        result = predict_emotion(image, model, device)

    with col2:
        st.subheader(f"Prediction: {result['predicted_class'].capitalize()}")
        st.write(f"Confidence: {result['confidence'] * 100:.1f}%")

        # Show all 7 class probabilities, sorted highest to lowest
        sorted_probs = dict(
            sorted(result["all_probabilities"].items(), key=lambda item: -item[1])
        )
        st.bar_chart(sorted_probs)
else:
    st.info("Upload or capture a photo above to get a prediction.")
