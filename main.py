import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from utils import preprocess_image, make_gradcam_heatmap, superimpose_heatmap


model = load_model("pneumonia_model.keras", custom_objects={'EfficientNetB0': EfficientNetB0})

class_names = ['NORMAL', 'PNEUMONIA']  


st.title("Pneumonia Detector with Grad-CAM")
st.write("Upload a chest X-ray image and let the AI predict and visualize the decision.")

uploaded_file = st.file_uploader("Choose a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image_file = Image.open(uploaded_file).convert('RGB')
    st.image(image_file, caption="Uploaded Chest X-ray", use_container_width=True)

    # Preprocess image
    img_array = preprocess_image(image_file)

    # Predict
    preds = model.predict(img_array)[0][0]  
    if preds > 0.5:
       pred_class = "PNEUMONIA"
    else:
       pred_class = "NORMAL"

    st.subheader(f"Prediction: **{pred_class}**")
    st.write("### Class Probabilities:")
    st.write(f"NORMAL: **{(1 - preds) * 100:.2f}%**")
    st.write(f"PNEUMONIA: **{preds * 100:.2f}%**")

    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, 'top_conv')
    superimposed_img = superimpose_heatmap(heatmap, image_file)

    st.subheader("Grad-CAM Visualization:")
    st.image(superimposed_img, caption="Model's Attention (Grad-CAM)", use_container_width=True)
